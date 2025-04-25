import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (Embedding, LSTM, Dense, Bidirectional,
                                     LayerNormalization, Dropout, Attention,
                                     Conv1D, Input, Concatenate)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint,
                                        ReduceLROnPlateau, Callback)
from tensorflow.keras import backend as K
import music21
from collections import defaultdict
import pickle
import random
import os
import glob
import argparse
from matplotlib import pyplot as plt


# ====================== 辅助函数 ======================
def load_midi_files(input_dir="input_midi", min_notes=50):
    """加载MIDI文件并返回有效的音符序列"""
    print(f"从目录 {input_dir} 加载MIDI文件...")
    preprocessor = EnhancedMidiPreprocessor(min_notes=min_notes)
    valid_sequences = []
    skipped_files = defaultdict(list)

    if not os.path.exists(input_dir):
        print(f"错误: 输入目录 {input_dir} 不存在")
        return [], {"目录不存在": input_dir}

    midi_patterns = ["*.mid", "*.midi", "*.MID", "*.MIDI"]
    file_list = []
    for pattern in midi_patterns:
        file_list.extend(glob.glob(os.path.join(input_dir, pattern)))

    print(f"找到 {len(file_list)} 个MIDI文件")

    for filepath in file_list:
        notes = preprocessor.parse_midi(filepath)
        if notes:
            valid_sequences.append(notes)
        else:
            skipped_files[os.path.basename(filepath)] = "音符数量不足或解析失败"

    print(f"加载完成: 有效序列={len(valid_sequences)}, 跳过={len(skipped_files)}")
    return valid_sequences, skipped_files


def create_example_data(output_dir):
    """创建示例训练数据"""
    print("创建示例训练数据...")
    os.makedirs(output_dir, exist_ok=True)
    example_file = os.path.join(output_dir, "example.mid")

    # 创建包含200个音符的简单旋律
    s = music21.stream.Stream()
    pitches = [60, 62, 64, 65, 67, 69, 71, 72]  # C大调音阶
    for i in range(200):
        n = music21.note.Note(pitches[i % 8])
        n.duration.quarterLength = 0.5
        s.append(n)

    s.write('midi', example_file)
    print(f"示例文件已创建: {example_file}")

    # 加载创建的示例文件
    sequences, _ = load_midi_files(output_dir)
    return sequences


# ====================== 音乐数据处理模块 ======================
class EnhancedMidiPreprocessor:
    def __init__(self, min_notes=50):
        self.min_notes  = min_notes
        self.note_to_int  = {}
        self.int_to_note  = {}
        self.duration_bins  = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]

    def quantize_duration(self, duration):
        """时值量化方法"""
        return min(self.duration_bins, key=lambda x: abs(x - duration))

    def parse_midi(self, filepath):
        """MIDI解析方法，提取更多音乐特征"""
        try:
            score = music21.converter.parse(filepath)
            notes = []

            # 提取全局音乐特征
            key = score.analyze('key') if len(score.flat.notes) > 10 else music21.key.Key('C')
            tempo = score.metronomeMarkBoundaries()[0][2].number

            for element in score.flatten().notesAndRests:
                # 跳过休止符和无效元素
                if element.isRest or not hasattr(element, 'duration'):
                    continue

                    # 和弦处理
                if element.isChord:
                    chord_type = element.chordKind if hasattr(element, 'chordKind') else 'unknown'
                    root = element.root().midi
                    pitches = f"chord_{root}_{chord_type}"
                else:
                    pitches = str(element.pitch.midi)

                    # 量化特征
                duration = self.quantize_duration(element.duration.quarterLength)
                velocity = element.volume.velocity if hasattr(element, 'volume') else 90
                offset = round(float(element.offset), 2)
                beat = int(element.beat) if hasattr(element, 'beat') else 1

                # 添加调性和节奏上下文
                encoded = (f"{pitches}_d{duration}_v{velocity}_"
                           f"b{beat}_k{key.tonic.name}_t{tempo}_o{offset}")
                notes.append(encoded)

            return notes if len(notes) >= self.min_notes else None

        except Exception as e:
            print(f"解析文件 {filepath} 时出错: {str(e)}")
            return None

    def create_vocabulary(self, note_sequences):
        """构建音符字典"""
        all_notes = [note for seq in note_sequences for note in seq]
        unique_notes = sorted(set(all_notes))

        self.note_to_int = {note: i for i, note in enumerate(unique_notes)}
        self.int_to_note = {i: note for i, note in enumerate(unique_notes)}

        # 保存字典
        with open('enhanced_vocabulary.pkl', 'wb') as f:
            pickle.dump((self.note_to_int, self.int_to_note), f)

    def prepare_sequences(self, sequences, seq_length=100):
        """准备训练序列，确保所有可能的音符都在词汇表中"""
        # 先收集所有可能的音符（包括增强生成的）
        all_possible_notes = set()

        for notes in sequences:
            for i in range(len(notes) - seq_length):
                seq_in = notes[i:i + seq_length]
                seq_out = notes[i + seq_length]

                # 收集基础音符
                all_possible_notes.update(seq_in + [seq_out])

                # 收集增强生成的可能音符
                for semitones in range(-3, 4):  # 覆盖所有可能的转调
                    transposed_in = [self._transpose_note(n, semitones) for n in seq_in]
                    transposed_out = self._transpose_note(seq_out, semitones)
                    all_possible_notes.update(transposed_in + [transposed_out])

        # 更新词汇表
        new_notes = all_possible_notes - set(self.note_to_int.keys())
        if new_notes:
            print(f"发现 {len(new_notes)} 个新音符，更新词汇表...")
            current_max = len(self.note_to_int)
            self.note_to_int.update({note: i + current_max for i, note in enumerate(new_notes)})
            self.int_to_note.update({i + current_max: note for i, note in enumerate(new_notes)})

        # 现在安全地生成训练序列
        X, y = [], []
        for notes in sequences:
            for i in range(len(notes) - seq_length):
                seq_in = notes[i:i + seq_length]
                seq_out = notes[i + seq_length]

                if random.random() > 0.7:
                    semitones = random.randint(-3, 3)
                    seq_in = [self._transpose_note(n, semitones) for n in seq_in]
                    seq_out = self._transpose_note(seq_out, semitones)

                X.append([self.note_to_int[note] for note in seq_in])
                y.append(self.note_to_int[seq_out])

        return np.array(X), np.array(y)

    def _transpose_note(self, note_str, semitones):
        """辅助方法：音符转调"""
        if note_str.startswith('chord_'):
            parts = note_str.split('_')
            root = int(parts[1]) + semitones
            return f"chord_{root}_{'_'.join(parts[2:])}"
        else:
            pitch = int(note_str.split('_')[0]) + semitones
            return f"{pitch}_{'_'.join(note_str.split('_')[1:])}"

        # ====================== 混合音乐模型 ======================


class EnhancedHybridMusicModel:
    def __init__(self, vocab_size, seq_length=100, preprocessor=None):
        self.vocab_size  = vocab_size
        self.seq_length  = seq_length
        self.preprocessor  = preprocessor
        self.model  = self._build_enhanced_model()

    def _build_enhanced_model(self):
        """修正后的增强模型架构"""
        # 音高输入
        pitch_input = Input(shape=(self.seq_length,), name='pitch_input')

        # 音高处理分支
        x = Embedding(self.vocab_size, 128)(pitch_input)
        x = Bidirectional(LSTM(256, return_sequences=True))(x)
        x = LayerNormalization()(x)
        x = Dropout(0.3)(x)
        pitch_features = Bidirectional(LSTM(128))(x)

        # 节奏输入
        rhythm_input = Input(shape=(self.seq_length, 2), name='rhythm_input')

        # 节奏处理分支
        y = Conv1D(64, 3, activation='relu', padding='same')(rhythm_input)
        y = Bidirectional(LSTM(64, return_sequences=True))(y)
        y = LayerNormalization()(y)
        y = Dropout(0.2)(y)
        rhythm_features = LSTM(32)(y)

        # 特征融合
        combined = Concatenate()([pitch_features, rhythm_features])
        combined = Dense(256, activation='relu')(combined)
        combined = Dropout(0.3)(combined)

        # 修正后的注意力层
        query = Dense(256)(combined)  # 关键修改：输出维度与combined一致
        attention = Attention()([query, combined])  # 现在维度匹配

        # 输出层
        pitch_output = Dense(self.vocab_size,
                             activation='softmax',
                             name='pitch_output')(attention)
        duration_output = Dense(1,
                                activation='linear',
                                name='duration_output')(attention)

        # 构建并编译模型
        model = Model(
            inputs=[pitch_input, rhythm_input],
            outputs=[pitch_output, duration_output]
        )

        optimizer = Adam(
            learning_rate=0.0005,
            clipnorm=1.0
        )

        model.compile(
            optimizer=optimizer,
            loss={
                'pitch_output': 'sparse_categorical_crossentropy',
                'duration_output': 'mse'
            },
            loss_weights={
                'pitch_output': 0.7,
                'duration_output': 0.3
            },
            metrics={
                'pitch_output': ['accuracy'],
                'duration_output': ['mae']
            }
        )

        return model

    def train(self, X, y, epochs=200, batch_size=32):
        X_rhythm = self._extract_enhanced_rhythm_features(X)
        y_duration = self._extract_duration_targets(y)

        callbacks = [
            EarlyStopping(monitor='val_pitch_output_accuracy', patience=50),
            ModelCheckpoint('best_model.h5', save_best_only=True),
            ReduceLROnPlateau(monitor='val_pitch_output_accuracy', patience=20)
        ]

        history = self.model.fit(
            [X, X_rhythm],
            {'pitch_output': y, 'duration_output': y_duration},
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.3,
            callbacks=callbacks
        )

        return history

    def _extract_enhanced_rhythm_features(self, sequences):
        rhythm_features = np.zeros((len(sequences),  self.seq_length,  2))

        for i, seq in enumerate(sequences):
            for j, note_idx in enumerate(seq):
                note_str = self.preprocessor.int_to_note[note_idx]
                parts = note_str.split('_')
                duration = float(parts[1][1:])
                beat_pos = int(parts[3][1:]) % 4
                rhythm_features[i, j] = [duration, beat_pos]

        return rhythm_features

    def _extract_duration_targets(self, y):
        """持续时间目标提取方法保持不变"""
        return np.array([
            float(self.preprocessor.int_to_note[idx].split('_')[1][1:])
            for idx in y
        ])
    # ====================== 改进版音乐生成器 ======================


class EnhancedMusicGenerator:
    def __init__(self, model, preprocessor, seq_length=100):
        self.model = model
        self.preprocessor = preprocessor
        self.seq_length = seq_length
        self.temperature = 0.7
        self.diversity_factors = {
            'temperature_decay': 0.99,
            'max_variation': 0.2
        }

    def _extract_rhythm_features(self, sequences):
        """从序列中提取节奏特征"""
        rhythm_features = np.zeros((len(sequences), self.seq_length, 2))

        for i, seq in enumerate(sequences):
            for j, note_idx in enumerate(seq):
                note_str = self.preprocessor.int_to_note[note_idx]
                parts = note_str.split('_')

                # 提取持续时间特征
                duration = float(parts[1][1:]) if len(parts) > 1 else 0.5
                # 提取节拍位置特征
                beat_pos = int(parts[3][1:]) % 4 if len(parts) > 3 else 0

                rhythm_features[i, j] = [duration, beat_pos]

        return rhythm_features

    def generate(self, seed_sequence=None, length=500, temperature=0.7):
        """改进的音乐生成方法"""
        if temperature < 0.1 or temperature > 1.5:
            temperature = 0.7

        if seed_sequence is None:
            # 选择有音乐性的起始音符
            chord_notes = [k for k in self.preprocessor.note_to_int.keys() if 'chord' in k]
            if chord_notes:
                seed_note = random.choice(chord_notes)
            else:
                seed_note = random.choice(list(self.preprocessor.note_to_int.keys()))

            seed_sequence = [self.preprocessor.note_to_int[seed_note]] * self.seq_length

        generated = list(seed_sequence)
        current_temp = temperature

        print(f"生成音乐 (初始温度={current_temp:.2f})...")
        for i in range(length):
            # 动态调整温度
            if i % 50 == 0:
                current_temp *= self.diversity_factors['temperature_decay']
                current_temp = max(current_temp, 0.1)
                print(f"进度: {i}/{length}, 当前温度: {current_temp:.2f}")

            # 准备当前序列
            current_seq = np.array([generated[-self.seq_length:]])
            rhythm_features = self._extract_rhythm_features(current_seq)

            # 预测下一个音符
            predictions = self.model.predict([current_seq, rhythm_features], verbose=0)
            pitch_probs, duration_pred = predictions

            # 应用温度采样
            next_pitch = self._temperature_sampling(pitch_probs[0], current_temp)

            # 添加音乐性约束
            if self._is_musically_invalid(next_pitch, generated):
                next_pitch = self._correct_pitch(generated)

            generated.append(next_pitch)

        print("生成完成")
        return [self.preprocessor.int_to_note[i] for i in generated]

    def _temperature_sampling(self, probs, temperature):
        """温度采样方法"""
        probs = np.log(probs) / temperature
        probs = np.exp(probs) / np.sum(np.exp(probs))
        return np.random.choice(len(probs), p=probs)

    def _is_musically_invalid(self, next_pitch, generated):
        """简单的音乐性验证"""
        # 避免太多重复音符
        if len(generated) > 10 and len(set(generated[-10:])) < 3:
            return True
        return False

    def _correct_pitch(self, generated):
        """音乐性修正"""
        last_pitch = generated[-1]
        vocab_size = len(self.preprocessor.note_to_int)  # 使用字典长度作为词汇表大小
        candidates = [
            (last_pitch + 3) % vocab_size,  # 三度音程
            (last_pitch + 4) % vocab_size,  # 大三度
            (last_pitch + 7) % vocab_size  # 五度音程
        ]
        return random.choice(candidates)
        # ====================== 实用工具和主流程 ======================


class MusicGenerationCallback(Callback):
    """训练期间生成样本的自定义回调"""

    def __init__(self, preprocessor):
        super().__init__()
        self.preprocessor = preprocessor

    def on_epoch_end(self, epoch, logs=None):
        if epoch % 10 == 0:
            print(f"\nEpoch {epoch} 训练中生成的样本:")
            sample = self._generate_sample()
            print(sample[:20])

    def _generate_sample(self):
        """生成简短样本"""
        model = self.model
        preprocessor = self.preprocessor

        seed = random.choice(list(preprocessor.note_to_int.keys()))
        seed_seq = [preprocessor.note_to_int[seed]] * model.input_shape[0][1]

        generated = []
        for _ in range(50):
            current_seq = np.array([seed_seq[-model.input_shape[0][1]:]])
            rhythm_features = np.zeros((1, model.input_shape[0][1], 2))

            preds = model.predict([current_seq, rhythm_features], verbose=0)
            next_pitch = np.argmax(preds[0][0])

            seed_seq.append(next_pitch)
            generated.append(preprocessor.int_to_note[next_pitch])

        return generated


def save_to_midi(note_sequence, output_path):
    """改进的MIDI保存方法"""
    try:
        stream = music21.stream.Stream()

        # 添加乐器信息
        piano = music21.instrument.Piano()
        stream.append(piano)

        # 解析并添加音符
        for note_str in note_sequence:
            parts = note_str.split('_')

            if parts[0].startswith('chord'):
                # 和弦处理
                root = int(parts[0].split('_')[1])
                chord = music21.chord.Chord([root, root + 4, root + 7])  # 大三和弦
                chord.duration.quarterLength = float(parts[1][1:])
                stream.append(chord)
            else:
                # 单音处理
                note = music21.note.Note(int(parts[0]))
                note.duration.quarterLength = float(parts[1][1:])
                note.volume.velocity = int(parts[2][1:])
                stream.append(note)

                # 设置速度和调性
        stream.insert(0, music21.tempo.MetronomeMark(number=120))

        # 尝试多种保存方式
        try:
            stream.write('midi', fp=output_path)
        except:
            alt_path = os.path.join(os.path.expanduser("~"), "Desktop", os.path.basename(output_path))
            stream.write('midi', fp=alt_path)
            print(f"无法写入原路径，已保存到桌面: {alt_path}")

    except Exception as e:
        print(f"保存MIDI文件时出错: {str(e)}")


if __name__ == "__main__":
    print("=== AI音乐生成系统启动 ===")

    # 参数解析
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="input_midi", help="MIDI输入目录")
    parser.add_argument("--output", default="enhanced_output.mid", help="输出MIDI文件")
    parser.add_argument("--seq_len", type=int, default=100, help="序列长度")
    parser.add_argument("--epochs", type=int, default=50, help="训练轮数")
    parser.add_argument("--temperature", type=float, default=0.7, help="生成温度")
    args = parser.parse_args()

    print("运行参数:")
    print(f"- 输入目录: {args.input}")
    print(f"- 输出文件: {args.output}")
    print(f"- 序列长度: {args.seq_len}")
    print(f"- 训练轮数: {args.epochs}")
    print(f"- 温度参数: {args.temperature}")

    # 1. 数据加载和预处理
    print("\n[阶段1] 加载和预处理MIDI文件...")
    preprocessor = EnhancedMidiPreprocessor(min_notes=50)
    sequences, _ = load_midi_files(args.input, min_notes=50)

    if not sequences:
        print("创建示例训练数据...")
        sequences = create_example_data(args.input)
        if not sequences:
            print("无法创建或加载训练数据，程序终止")
            exit(1)

    # 先创建基础词汇表
    preprocessor.create_vocabulary(sequences)

    # 在准备序列前确保词汇表完整
    print("准备训练序列并确保词汇表完整...")
    X, y = preprocessor.prepare_sequences(sequences, args.seq_len)

    # 检查词汇表大小
    print(f"最终词汇表大小: {len(preprocessor.note_to_int)}")

    # 2. 模型训练
    print("\n[阶段2] 模型训练...")
    model = EnhancedHybridMusicModel(
        vocab_size=len(preprocessor.note_to_int),
        seq_length=args.seq_len,
        preprocessor=preprocessor
    )

    history = model.train(X, y, epochs=args.epochs)

    # 3. 音乐生成
    print("\n[阶段3] 音乐生成...")
    generator = EnhancedMusicGenerator(
        model=model.model,
        preprocessor=preprocessor,
        seq_length=args.seq_len
    )

    generated = generator.generate(length=200, temperature=args.temperature)

    # 4. 保存结果
    print("\n[阶段4] 保存结果...")
    save_to_midi(generated, args.output)
    print(f"\n=== 完成! 生成结果已保存至 {args.output}  ===")