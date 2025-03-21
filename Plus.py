import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (Input, Embedding, Dense, Dropout,
                                     LayerNormalization, MultiHeadAttention,
                                     Concatenate, Flatten)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from music21 import converter, instrument, note, chord, stream
import pretty_midi
import math


# 一、强化版MIDI解析器（支持多乐器轨道与异常处理）
class AdvancedMIDIParser:
    def __init__(self, quantization_step=0.125):
        self.quant_step = quantization_step  # 三十二分音符级精度量化

    def parse(self, file_path):
        midi_data = converter.parse(file_path)
        feature_matrix = {
            'notes': [],
            'durations': [],
            'offsets': [],
            'velocities': [],
            'instruments': [],
            'chords': []
        }

        for part in midi_data.parts:
            instrument_name = part.partName or "Piano"
            for event in part.flatten().notesAndRests:  # 使用新API替代弃用方法
                quantized_offset = self._quantize_value(event.offset)
                quantized_duration = self._quantize_value(event.duration.quarterLength)

                # 增强型事件处理逻辑
                if isinstance(event, note.Note):
                    self._process_note(event, feature_matrix, instrument_name, quantized_duration, quantized_offset)
                elif isinstance(event, chord.Chord):
                    self._process_chord(event, feature_matrix, instrument_name, quantized_duration, quantized_offset)
                else:
                    self._process_rest(feature_matrix, quantized_duration, quantized_offset, instrument_name)

        return feature_matrix

    def _quantize_value(self, value):
        """音乐理论约束的量化函数"""
        return round(value / self.quant_step) * self.quant_step

    def _process_note(self, event, matrix, instr, dur, offset):
        """处理单音符事件"""
        velocity = getattr(event.volume, 'velocity', 80)
        matrix['notes'].append(event.pitch.midi)
        matrix['durations'].append(dur)
        matrix['offsets'].append(offset)
        matrix['velocities'].append(velocity)
        matrix['instruments'].append(instr)

    def _process_chord(self, event, matrix, instr, dur, offset):
        """处理和弦事件"""
        for n in event.notes:
            matrix['notes'].append(n.pitch.midi)
            matrix['durations'].append(dur)
            matrix['offsets'].append(offset)
            matrix['velocities'].append(80)  # 和弦统一速度
            matrix['instruments'].append(instr)
        matrix['chords'].append(tuple(n.pitch.midi for n in event.notes))

    def _process_rest(self, matrix, dur, offset, instr):
        """处理休止符事件"""
        matrix['notes'].append(0)
        matrix['durations'].append(dur)
        matrix['offsets'].append(offset)
        matrix['velocities'].append(0)
        matrix['instruments'].append(instr)  # 保留原始乐器名称

    # 二、音乐理论编码系统


class MusicTheoryEncoder:
    def __init__(self, features):
        self.note_vocab = self._build_note_vocab(features['notes'])
        self.duration_bins = self._quantize_durations(features['durations'])
        self.offset_bins = self._quantize_offsets(features['offsets'])

    def _build_note_vocab(self, notes):
        unique_notes = sorted(set(notes) | {0, 128})  # 添加休止符和延音符号
        return {n: i for i, n in enumerate(unique_notes)}

    def _quantize_durations(self, durations):
        # 音乐理论约束的时值量化
        base_values = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]
        return np.digitize(durations, base_values)

    def _quantize_offsets(self, offsets):
        # 小节对齐量化
        max_offset = math.ceil(max(offsets))
        bar_length = 4  # 4/4拍基准
        return np.digitize(offsets, np.arange(0, max_offset, bar_length))


# 三、Transformer模型架构（带掩码机制）
def create_transformer_model(vocab_size, num_durations, num_offsets,
                             embed_dim=512, num_heads=8, ff_dim=2048, num_layers=6):
    # 多模态输入定义
    note_input = Input(shape=(None,), name='note_input', dtype=tf.int32)
    duration_input = Input(shape=(None,), name='duration_input', dtype=tf.int32)
    offset_input = Input(shape=(None,), name='offset_input', dtype=tf.int32)

    # 特征嵌入系统
    note_emb = Embedding(vocab_size, embed_dim)(note_input)
    duration_emb = Embedding(num_durations, embed_dim // 4)(duration_input)
    offset_emb = Embedding(num_offsets, embed_dim // 4)(offset_input)

    # 动态投影融合
    combined = Concatenate()([note_emb, duration_emb, offset_emb])
    x = Dense(embed_dim, activation='gelu')(combined)

    # 位置编码层
    x = PositionalEncoding(embed_dim)(x)

    # Transformer层堆叠
    for _ in range(num_layers):
        x = TransformerBlock(embed_dim, num_heads, ff_dim)(x)

    # 多任务预测头
    note_output = Dense(vocab_size, activation='softmax', name='note_output')(x)
    duration_output = Dense(num_durations, activation='softmax', name='duration_output')(x)
    offset_output = Dense(num_offsets, activation='softmax', name='offset_output')(x)

    return Model(inputs=[note_input, duration_input, offset_input],
                 outputs=[note_output, duration_output, offset_output])


class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim):
        super().__init__()
        self.attn = MultiHeadAttention(num_heads, embed_dim)
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation="gelu"),
            Dense(embed_dim)
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout = Dropout(0.1)

    def call(self, inputs):
        # 自注意力掩码机制
        seq_len = tf.shape(inputs)[1]
        look_ahead_mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)

        attn_output = self.attn(inputs, inputs, attention_mask=look_ahead_mask)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        return self.layernorm2(out1 + ffn_output)


class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model

    def call(self, inputs):
        seq_length = tf.shape(inputs)[1]
        position = tf.cast(tf.range(seq_length), tf.float32)
        div_term = tf.exp(tf.range(0, self.d_model, 2, dtype=tf.float32) *
                          (-math.log(10000.0) / self.d_model))
        sin_part = tf.sin(position[:, tf.newaxis] * div_term)
        cos_part = tf.cos(position[:, tf.newaxis] * div_term)
        pos_encoding = tf.concat([sin_part, cos_part], axis=-1)
        pos_encoding = tf.expand_dims(pos_encoding, 0)
        return inputs + pos_encoding

    # 四、训练系统（带课程学习策略）


class CurriculumTrainer:
    def __init__(self, model, init_seq_len=32, max_seq_len=512):
        self.model = model
        self.seq_len = init_seq_len
        self.max_seq_len = max_seq_len

    def dynamic_batching(self, dataset):
        # 动态序列长度调整策略
        def _adjust_length(record):
            return record[:, :self.seq_len]

        return dataset.map(_adjust_length)

    def progressive_training(self, dataset, epochs=100):
        for epoch in range(epochs):
            # 每10个epoch扩展序列长度
            if epoch % 10 == 0 and self.seq_len < self.max_seq_len:
                self.seq_len = min(self.seq_len * 2, self.max_seq_len)
                # 训练步骤（此处为示例框架）
                self.model.fit(dataset, epochs=1)


# 五、音乐生成器（带风格控制）
class IntelligentComposer:
    def __init__(self, model, tokenizer, temp_schedule_fn=None):
        self.model = model
        self.tokenizer = tokenizer
        self.temp_schedule = temp_schedule_fn or (lambda step: 0.5 + 0.3 * math.sin(step / 10))

    def generate(self, seed_sequence, length=512):
        generated = seed_sequence.copy()
        for step in range(length):
            # 动态温度调节
            temp = self.temp_schedule(step)
            # 带掩码的预测
            predictions = self.model.predict(generated[-self.seq_len:])
            sampled_note = self._sample_with_temp(predictions[0][-1], temp)
            # 音乐理论约束
            if self._is_rest(sampled_note):
                generated.append(self._apply_rest_rules())
            else:
                generated.append(sampled_note)
        return generated

    def _sample_with_temp(self, logits, temperature):
        logits = logits / temperature
        return tf.random.categorical(logits, num_samples=1)


# 六、部署配置
# 六、部署配置（增加异常处理模块）
if __name__ == "__main__":
    # GPU加速配置（适配TensorFlow 2.15+）
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.set_logical_device_configuration(
                gpus[0],
                [tf.config.LogicalDeviceConfiguration(memory_limit=6144)]
            )
        except RuntimeError as e:
            print(f"GPU配置错误: {e}")

    # 增强型数据流水线
    try:
        parser = AdvancedMIDIParser()
        features = parser.parse("input.mid")
        encoder = MusicTheoryEncoder(features)
    except Exception as e:
        print(f"MIDI解析失败: {str(e)}")
        exit(1)

if __name__ == "__main__":
    # 硬件加速配置
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    # 超参数设置
    model_config = {
        'embed_dim': 512,
        'num_heads': 8,
        'ff_dim': 2048,
        'num_layers': 8
    }

    train_config = {
        'learning_rate': 0.0001,
        'batch_size': 64,
        'max_seq_len': 512
    }



    # 模型构建（修正后）
    transformer = create_transformer_model(
        vocab_size=len(encoder.note_vocab),
        num_durations=encoder.duration_bins,
        num_offsets=encoder.offset_bins,
        **model_config  # 仅传递架构参数
    )

    # 优化器配置（独立设置）
    optimizer = Adam(
        learning_rate=train_config['learning_rate'],
        clipnorm=1.0,
        clipvalue=0.5,
        global_clipnorm=2.0  # 新增梯度约束
    )

    # 数据流水线
    parser = AdvancedMIDIParser()
    features = parser.parse("input.mid")
    encoder = MusicTheoryEncoder(features)

    # 模型构建
    transformer = create_transformer_model(
        vocab_size=len(encoder.note_vocab),
        num_durations=encoder.duration_bins,
        num_offsets=encoder.offset_bins,
        **training_config
    )

    # 优化系统
    optimizer = Adam(
        learning_rate=training_config['learning_rate'],
        clipnorm=1.0,
        clipvalue=0.5
    )
    transformer.compile(optimizer=optimizer,
                        loss='sparse_categorical_crossentropy',
                        loss_weights=[0.6, 0.3, 0.1])

    # 训练执行
    trainer = CurriculumTrainer(transformer)
    dataset = ...  # 数据预处理流程
    trainer.progressive_training(dataset)

    # 生成演示 
    composer = IntelligentComposer(transformer, encoder)
    generated_music = composer.generate(seed_sequence=[...])

    #日志记录
    import json

    print("Model Config:", json.dumps(model_config, indent=2))
    print("Train Config:", json.dumps(train_config, indent=2))