import numpy as np 
import tensorflow as tf 
from tensorflow.keras.layers  import (Embedding, LSTM, Dense, Bidirectional,
                                     LayerNormalization, Dropout, Attention,
                                     Conv1D, Input, Concatenate)
from tensorflow.keras.models  import Model 
from tensorflow.keras.optimizers  import Adam 
from tensorflow.keras.callbacks  import (EarlyStopping, ModelCheckpoint,
                                        ReduceLROnPlateau, Callback)
from tensorflow.keras  import backend as K 
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
        file_list.extend(glob.glob(os.path.join(input_dir,  pattern)))
 
    print(f"找到 {len(file_list)} 个MIDI文件")
 
    for filepath in file_list:
        try:
            notes = preprocessor.parse_midi(filepath) 
            if notes:
                valid_sequences.append(notes) 
            else:
                skipped_files[os.path.basename(filepath)] = "音符数量不足或解析失败"
        except Exception as e:
            skipped_files[os.path.basename(filepath)] = f"解析错误: {str(e)}"
            continue 
 
    print(f"加载完成: 有效序列={len(valid_sequences)}, 跳过={len(skipped_files)}")
    return valid_sequences, skipped_files 
 
def create_example_data(output_dir):
    """创建示例训练数据"""
    print("创建示例训练数据...")
    os.makedirs(output_dir,  exist_ok=True)
    example_file = os.path.join(output_dir,  "example.mid") 
 
    # 创建包含200个音符的简单旋律 
    s = music21.stream.Stream() 
    pitches = [60, 62, 64, 65, 67, 69, 71, 72]  # C大调音阶 
    for i in range(200):
        n = music21.note.Note(pitches[i  % 8])
        n.duration.quarterLength  = 0.5 + random.random()  * 0.5  # 随机时长增加变化 
        s.append(n) 
 
    s.write('midi',  example_file)
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
        self.training_sequences  = []  # 存储训练序列用于种子生成 
 
    def quantize_duration(self, duration):
        """时值量化方法"""
        return min(self.duration_bins,  key=lambda x: abs(x - duration))
 
    def parse_midi(self, filepath):
        """MIDI解析方法，提取更多音乐特征"""
        try:
            score = music21.converter.parse(filepath) 
            notes = []
 
            # 提取全局音乐特征 
            key = score.analyze('key')  if len(score.flat.notes)  > 10 else music21.key.Key('C') 
            tempo = score.metronomeMarkBoundaries()[0][2].number  if score.metronomeMarkBoundaries()  else 120 
 
            for element in score.flatten().notesAndRests: 
                if element.isRest  or not hasattr(element, 'duration'):
                    continue 
 
                # 和弦处理 
                if element.isChord: 
                    chord_type = element.chordKind  if hasattr(element, 'chordKind') else 'major'
                    root = element.root().midi  
                    pitches = f"chord_{root}_{chord_type}"
                else:
                    pitches = str(element.pitch.midi) 
 
                # 量化特征 
                duration = self.quantize_duration(element.duration.quarterLength) 
                velocity = element.volume.velocity  if hasattr(element, 'volume') else 90 
                offset = round(float(element.offset),  2)
                beat = int(element.beat)  if hasattr(element, 'beat') else 1 
 
                # 添加调性和节奏上下文 
                encoded = (f"{pitches}_d{duration}_v{velocity}_"
                         f"b{beat}_k{key.tonic.name}_t{tempo}_o{offset}") 
                notes.append(encoded) 
 
            return notes if len(notes) >= self.min_notes  else None 
 
        except Exception as e:
            print(f"解析文件 {filepath} 时出错: {str(e)}")
            return None 
 
    def create_vocabulary(self, note_sequences):
        """词汇表创建方法"""
        all_notes = [note for seq in note_sequences for note in seq]
        
        # 添加数据增强可能生成的所有音符变体 
        enhanced_notes = set(all_notes)
        for note in all_notes:
            for semitones in range(-5, 6):  # 覆盖所有可能的转调 
                enhanced_notes.add(self._transpose_note(note,  semitones))
            # 添加节奏变体 
            enhanced_notes.add(self._vary_rhythm(note)) 
        
        unique_notes = sorted(enhanced_notes)
        self.note_to_int  = {note: i for i, note in enumerate(unique_notes)}
        self.int_to_note  = {i: note for i, note in enumerate(unique_notes)}
        
        # 保存字典 
        with open('enhanced_vocabulary.pkl',  'wb') as f:
            pickle.dump((self.note_to_int,  self.int_to_note),  f)
        
        print(f"词汇表创建完成，包含 {len(self.note_to_int)}  个独特音符编码")
 
    def prepare_sequences(self, sequences, seq_length=100):
        """训练序列准备方法"""
        self.training_sequences  = sequences 
        
        # 首先确保词汇表包含所有可能的音符 
        self._ensure_vocabulary_completeness(sequences, seq_length)
        
        X, y = [], []
        for notes in sequences:
            for i in range(len(notes) - seq_length):
                seq_in = notes[i:i + seq_length]
                seq_out = notes[i + seq_length]
                
                # 50%概率进行数据增强 
                if random.random()  > 0.5:
                    semitones = random.randint(-5,  5)
                    try:
                        seq_in = [self._transpose_note(n, semitones) for n in seq_in]
                        seq_out = self._transpose_note(seq_out, semitones)
                    except Exception as e:
                        print(f"转调时出错: {str(e)}")
                        continue 
                    
                    if random.random()  > 0.7:
                        try:
                            seq_in = [self._vary_rhythm(n) for n in seq_in]
                            seq_out = self._vary_rhythm(seq_out)
                        except Exception as e:
                            print(f"变化节奏时出错: {str(e)}")
                            continue 
                
                try:
                    X.append([self.note_to_int[note]  for note in seq_in])
                    y.append(self.note_to_int[seq_out]) 
                except KeyError as e:
                    print(f"警告: 发现未处理的音符变体 {e}，已跳过该序列")
                    continue 
        
        return np.array(X),  np.array(y) 
 
    def _ensure_vocabulary_completeness(self, sequences, seq_length):
        """确保词汇表包含所有可能的音符变体"""
        new_notes = set()
        
        for notes in sequences:
            for i in range(len(notes) - seq_length):
                seq_in = notes[i:i + seq_length]
                seq_out = notes[i + seq_length]
                
                # 检查基础序列 
                for note in seq_in + [seq_out]:
                    if note not in self.note_to_int: 
                        new_notes.add(note) 
                
                # 检查可能的转调变体 
                for semitones in range(-5, 6):
                    transposed = self._transpose_note(seq_out, semitones)
                    if transposed not in self.note_to_int: 
                        new_notes.add(transposed) 
                
                # 检查节奏变体 
                rhythm_varied = self._vary_rhythm(seq_out)
                if rhythm_varied not in self.note_to_int: 
                    new_notes.add(rhythm_varied) 
        
        if new_notes:
            print(f"发现 {len(new_notes)} 个新音符，更新词汇表...")
            current_max = len(self.note_to_int) 
            self.note_to_int.update({note:  i + current_max for i, note in enumerate(new_notes)})
            self.int_to_note.update({i  + current_max: note for i, note in enumerate(new_notes)})
            
            # 更新保存的词汇表 
            with open('enhanced_vocabulary.pkl',  'wb') as f:
                pickle.dump((self.note_to_int,  self.int_to_note),  f)
 
    def _transpose_note(self, note_str, semitones):
        """音符转调"""
        if note_str.startswith('chord_'): 
            parts = note_str.split('_') 
            try:
                root = int(parts[1]) + semitones 
                return f"chord_{root}_{'_'.join(parts[2:])}"
            except (IndexError, ValueError):
                return note_str 
        else:
            try:
                parts = note_str.split('_') 
                pitch = int(parts[0]) + semitones 
                return f"{pitch}_{'_'.join(parts[1:])}"
            except (IndexError, ValueError):
                return note_str 
 
    def _vary_rhythm(self, note_str):
        """随机变化节奏"""
        parts = note_str.split('_') 
        if len(parts) > 1:
            try:
                orig_dur = float(parts[1][1:])
                new_dur = orig_dur * random.uniform(0.75,  1.25)
                parts[1] = f"d{self.quantize_duration(new_dur)}" 
                return "_".join(parts)
            except (IndexError, ValueError):
                return note_str 
        return note_str 
 
# ====================== 混合音乐模型 ======================
class EnhancedHybridMusicModel:
    def __init__(self, vocab_size, seq_length=100, preprocessor=None):
        self.vocab_size  = vocab_size 
        self.seq_length  = seq_length 
        self.preprocessor  = preprocessor 
        self.model  = self._build_enhanced_model()
 
    def _build_enhanced_model(self):
        """改进的模型架构"""
        # 音高输入 
        pitch_input = Input(shape=(self.seq_length,),  name='pitch_input')
        
        # 增强的音高嵌入 
        x = Embedding(self.vocab_size,  256)(pitch_input)
        x = Dropout(0.2)(x)
        
        # 双向LSTM层 
        x = Bidirectional(LSTM(512, return_sequences=True))(x)
        x = LayerNormalization()(x)
        x = Dropout(0.3)(x)
        
        # 添加卷积层捕捉局部模式 
        x = Conv1D(256, 5, activation='relu', padding='same')(x)
        x = LayerNormalization()(x)
        
        # 第二层LSTM 
        x = Bidirectional(LSTM(256))(x)
        x = LayerNormalization()(x)
        
        # 节奏输入处理 
        rhythm_input = Input(shape=(self.seq_length,  2), name='rhythm_input')
        y = Conv1D(128, 3, activation='relu', padding='same')(rhythm_input)
        y = Bidirectional(LSTM(128, return_sequences=True))(y)
        y = LayerNormalization()(y)
        y = LSTM(64)(y)
        
        # 特征融合 
        combined = Concatenate()([x, y])
        combined = Dense(512, activation='relu')(combined)
        combined = Dropout(0.4)(combined)
        
        # 输出层 
        pitch_output = Dense(self.vocab_size,  
                           activation='softmax', 
                           name='pitch_output')(combined)
        
        # 构建并编译模型 
        model = Model(
            inputs=[pitch_input, rhythm_input],
            outputs=pitch_output 
        )
        
        optimizer = Adam(learning_rate=0.0005, clipnorm=1.0)
        model.compile(optimizer=optimizer, 
                     loss='sparse_categorical_crossentropy',
                     metrics=['accuracy'])
        
        return model 
 
    def train(self, X, y, epochs=200, batch_size=32):
        X_rhythm = self._extract_enhanced_rhythm_features(X)
        
        callbacks = [
            EarlyStopping(monitor='val_accuracy', patience=50),
            ModelCheckpoint('best_model.h5', save_best_only=True),
            ReduceLROnPlateau(monitor='val_accuracy', patience=20)
        ]
 
        history = self.model.fit( 
            [X, X_rhythm],
            y,
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
                
                try:
                    duration = float(parts[1][1:]) if len(parts) > 1 else 0.5 
                    beat_pos = int(parts[3][1:]) % 4 if len(parts) > 3 else 0 
                except (IndexError, ValueError):
                    duration = 0.5 
                    beat_pos = 0 
                
                rhythm_features[i, j] = [duration, beat_pos]
 
        return rhythm_features 
 
# ====================== 音乐生成器 ======================
class EnhancedMusicGenerator:
    def __init__(self, model, preprocessor, seq_length=100):
        self.model  = model 
        self.preprocessor  = preprocessor 
        self.seq_length  = seq_length 
        self.temperature  = 0.7 
        self.diversity_factors  = {
            'temperature_decay': 0.99,
            'max_variation': 0.2 
        }
 
    def _extract_rhythm_features(self, sequences):
        """从序列中提取节奏特征"""
        rhythm_features = np.zeros((len(sequences),  self.seq_length,  2))
 
        for i, seq in enumerate(sequences):
            for j, note_idx in enumerate(seq):
                note_str = self.preprocessor.int_to_note[note_idx] 
                parts = note_str.split('_') 
 
                try:
                    duration = float(parts[1][1:]) if len(parts) > 1 else 0.5 
                    beat_pos = int(parts[3][1:]) % 4 if len(parts) > 3 else 0 
                except (IndexError, ValueError):
                    duration = 0.5 
                    beat_pos = 0 
 
                rhythm_features[i, j] = [duration, beat_pos]
 
        return rhythm_features 
 
    def generate(self, seed_sequence=None, length=500, temperature=0.7):
        """音乐生成方法"""
        if temperature < 0.1 or temperature > 1.5:
            temperature = 0.7 
 
        if seed_sequence is None:
            # 从训练数据中随机选取有音乐性的片段作为种子 
            if hasattr(self.preprocessor,  'training_sequences'):
                valid_seeds = [seq for seq in self.preprocessor.training_sequences  
                             if len(set(seq)) > 5]  # 选择有变化的序列 
                if valid_seeds:
                    seed_seq = random.choice(valid_seeds) 
                    seed_sequence = [self.preprocessor.note_to_int[note] 
                                   for note in seed_seq[-self.seq_length:]] 
                else:
                    # 创建和弦进行种子 
                    roots = [48, 50, 52, 53, 55, 57, 59, 60]  # C大调音阶 
                    chord_types = ['major', 'minor', 'diminished', 'augmented']
                    seed_sequence = [self.preprocessor.note_to_int[f"chord_{random.choice(roots)}_{random.choice(chord_types)}"] 
                                   for _ in range(self.seq_length)] 
            else:
                # 备用方案：创建音阶种子 
                scale_degrees = [60, 62, 64, 65, 67, 69, 71, 72]  # C大调 
                seed_sequence = [self.preprocessor.note_to_int[str(random.choice(scale_degrees))] 
                               for _ in range(self.seq_length)] 
 
        generated = list(seed_sequence)
        current_temp = temperature * 1.5  # 初始温度更高 
 
        print(f"生成音乐 (初始温度={current_temp:.2f})...")
        for i in range(length):
            # 动态调整温度 - 开头变化更大 
            if i < 50:  # 前50个音符 
                current_temp = max(temperature * (1.5 - i/100), temperature)
            else:
                if i % 50 == 0:
                    current_temp *= self.diversity_factors['temperature_decay'] 
                    current_temp = max(current_temp, temperature*0.8)
 
            current_seq = np.array([generated[-self.seq_length:]]) 
            rhythm_features = self._extract_rhythm_features(current_seq)
 
            predictions = self.model.predict([current_seq,  rhythm_features], verbose=0)
            pitch_probs = predictions[0] if isinstance(predictions, list) else predictions 
 
            next_pitch = self._temperature_sampling(pitch_probs[0], current_temp)
 
            # 添加音乐性约束 
            if self._is_musically_invalid(next_pitch, generated):
                next_pitch = self._correct_pitch(generated)
 
            generated.append(next_pitch) 
 
        print("生成完成")
        return [self.preprocessor.int_to_note[i] for i in generated]
 
    def _temperature_sampling(self, probs, temperature):
        """温度采样方法"""
        probs = np.log(probs  + 1e-10) / temperature  # 添加小值防止log(0)
        probs = np.exp(probs)  / np.sum(np.exp(probs)) 
        return np.random.choice(len(probs),  p=probs)
 
    def _is_musically_invalid(self, next_pitch, generated):
        """音乐性验证"""
        # 避免太多重复音符 
        if len(generated) > 10 and len(set(generated[-10:])) < 3:
            return True 
        
        # 避免不和谐的音程跳跃(超过八度)
        if len(generated) > 1:
            last_pitch = generated[-1]
            interval = abs(next_pitch - last_pitch)
            if interval > 12:  # 超过八度 
                return True 
        
        return False 
 
    def _correct_pitch(self, generated):
        """音高修正"""
        last_pitch = generated[-1]
        vocab_size = len(self.preprocessor.note_to_int) 
        
        # 基于音乐理论的候选音高 
        candidates = [
            (last_pitch + 2) % vocab_size,  # 大二度 
            (last_pitch + 4) % vocab_size,  # 大三度 
            (last_pitch + 5) % vocab_size,  # 纯四度 
            (last_pitch + 7) % vocab_size,  # 纯五度 
            (last_pitch - 2) % vocab_size,  # 下大二度 
            (last_pitch - 5) % vocab_size   # 下纯四度 
        ]
        
        # 优先选择和弦音 
        chord_notes = [k for k, v in self.preprocessor.note_to_int.items()  if 'chord' in k]
        if chord_notes:
            chord_root = int(random.choice(chord_notes).split('_')[1]) 
            candidates.extend([ 
                chord_root % vocab_size,
                (chord_root + 4) % vocab_size,
                (chord_root + 7) % vocab_size 
            ])
        
        return random.choice(candidates) 
 
# ====================== 实用工具和主流程 ======================
def save_to_midi(note_sequence, output_path):
    """MIDI保存方法"""
    try:
        stream = music21.stream.Stream() 
        piano = music21.instrument.Piano() 
        stream.append(piano) 
 
        for note_str in note_sequence:
            parts = note_str.split('_') 
 
            if parts[0].startswith('chord'):
                # 和弦处理 
                try:
                    root = int(parts[0].split('_')[1])
                    chord_type = parts[0].split('_')[2]
                    if chord_type == 'major':
                        chord = music21.chord.Chord([root,  root + 4, root + 7])
                    elif chord_type == 'minor':
                        chord = music21.chord.Chord([root,  root + 3, root + 7])
                    elif chord_type == 'diminished':
                        chord = music21.chord.Chord([root,  root + 3, root + 6])
                    elif chord_type == 'augmented':
                        chord = music21.chord.Chord([root,  root + 4, root + 8])
                    else:
                        chord = music21.chord.Chord([root,  root + 4, root + 7])
                    
                    duration = float(parts[1][1:]) if len(parts) > 1 else 0.5 
                    chord.duration.quarterLength  = duration 
                    stream.append(chord) 
                except (IndexError, ValueError):
                    continue 
            else:
                # 单音处理 
                try:
                    pitch = int(parts[0])
                    duration = float(parts[1][1:]) if len(parts) > 1 else 0.5 
                    velocity = int(parts[2][1:]) if len(parts) > 2 else 90 
                    
                    note = music21.note.Note(pitch) 
                    note.duration.quarterLength  = duration 
                    note.volume.velocity  = velocity 
                    stream.append(note) 
                except (IndexError, ValueError):
                    continue 
 
        # 设置速度和调性 
        stream.insert(0,  music21.tempo.MetronomeMark(number=120)) 
 
        try:
            stream.write('midi',  fp=output_path)
        except:
            alt_path = os.path.join(os.path.expanduser("~"),  "Desktop", os.path.basename(output_path)) 
            stream.write('midi',  fp=alt_path)
            print(f"无法写入原路径，已保存到桌面: {alt_path}")
 
    except Exception as e:
        print(f"保存MIDI文件时出错: {str(e)}")
 
if __name__ == "__main__":
    print("========Starting...========")
 
    # 参数解析 
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="input_midi", help="MIDI输入目录")
    parser.add_argument("--output",  default="enhanced_output.mid",  help="输出MIDI文件")
    parser.add_argument("--seq_len",  type=int, default=100, help="序列长度")
    parser.add_argument("--epochs",  type=int, default=20, help="训练轮数")
    parser.add_argument("--temperature",  type=float, default=0.7, help="生成温度")
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
    sequences, _ = load_midi_files(args.input,  min_notes=50)
 
    if not sequences:
        print("创建示例训练数据...")
        sequences = create_example_data(args.input) 
        if not sequences:
            print("无法创建或加载训练数据，程序终止")
            exit(1)
 
    # 2. 创建词汇表 
    print("\n[阶段2] 创建词汇表...")
    try:
        if os.path.exists('enhanced_vocabulary.pkl'): 
            with open('enhanced_vocabulary.pkl',  'rb') as f:
                preprocessor.note_to_int,  preprocessor.int_to_note  = pickle.load(f) 
            print(f"从文件加载词汇表，包含 {len(preprocessor.note_to_int)}  个音符编码")
        else:
            preprocessor.create_vocabulary(sequences) 
    except Exception as e:
        print(f"加载/创建词汇表时出错: {str(e)}")
        preprocessor.create_vocabulary(sequences) 
    
    # 3. 准备训练序列 
    print("\n[阶段3] 准备训练序列...")
    try:
        X, y = preprocessor.prepare_sequences(sequences,  args.seq_len) 
        print(f"训练数据形状: X={X.shape},  y={y.shape}") 
        print(f"词汇表大小: {len(preprocessor.note_to_int)}") 
    except Exception as e:
        print(f"准备训练序列时出错: {str(e)}")
        exit(1)
 
    # 4. 模型训练 
    print("\n[阶段4] 模型训练...")
    model = EnhancedHybridMusicModel(
        vocab_size=len(preprocessor.note_to_int), 
        seq_length=args.seq_len, 
        preprocessor=preprocessor 
    )
 
    try:
        history = model.train(X,  y, epochs=args.epochs) 
        print("训练完成")
    except Exception as e:
        print(f"模型训练时出错: {str(e)}")
        exit(1)
 
    # 5. 音乐生成 
    print("\n[阶段5] 音乐生成...")
    generator = EnhancedMusicGenerator(
        model=model.model, 
        preprocessor=preprocessor,
        seq_length=args.seq_len  
    )
 
    try:
        generated = generator.generate(length=200,  temperature=args.temperature) 
    except Exception as e:
        print(f"音乐生成时出错: {str(e)}")
        exit(1)
 
    # 6. 保存结果 
    print("\n[阶段6] 保存结果...")
    try:
        save_to_midi(generated, args.output) 
        print(f"\n=== 完成! 生成结果已保存至 {args.output}  ===")
    except Exception as e:
        print(f"保存结果时出错: {str(e)}")