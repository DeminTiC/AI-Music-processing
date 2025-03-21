# 环境依赖（需先安装下方列出的软件包）
import numpy as np
from music21 import converter, instrument, note, chord, stream
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Embedding
from tensorflow.keras.utils import to_categorical
import pretty_midi


# 1. MIDI数据处理模块
def load_midi_data(file_path):
    midi = converter.parse(file_path)
    notes = []
    for element in midi.flat.notes:
        if isinstance(element, note.Note):
            notes.append(str(element.pitch.midi))   # 使用MIDI编号
        elif isinstance(element, chord.Chord):
            notes.append('.'.join(str(n.pitch.midi)  for n in element.notes))   # 和弦MIDI编号
    return notes


# 2. 序列编码与数据集构建
def create_sequences(notes, sequence_length=100):
    unique_notes = sorted(set(notes))
    note_to_int = {note: idx for idx, note in enumerate(unique_notes)}

    network_input = []
    network_output = []
    for i in range(len(notes) - sequence_length):
        seq_in = notes[i:i + sequence_length]
        seq_out = notes[i + sequence_length]
        network_input.append([note_to_int[char] for char in seq_in])
        network_output.append(note_to_int[seq_out])

    return np.array(network_input), np.array(network_output), note_to_int


# 3. LSTM模型构建
def build_model(input_shape, num_unique_notes):
    model = Sequential([
        Embedding(input_dim=num_unique_notes, output_dim=64),
        LSTM(256, return_sequences=True),
        Dropout(0.3),
        LSTM(256),
        Dense(num_unique_notes, activation='softmax')
    ])
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')
    return model


# 4. 音乐生成算法
def generate_music(model, start_sequence, note_to_int, int_to_note, length=500, temperature=1.0):
    sequence_length = len(start_sequence)
    generated = start_sequence.copy()
    for _ in range(length):
        input_seq = np.array([note_to_int[n] for n in generated[-sequence_length:]])
        input_seq = input_seq.reshape(1, -1)

        predicted_probs = model.predict(input_seq, verbose=0)[0]
        predicted_probs = np.log(predicted_probs) / temperature
        exp_probs = np.exp(predicted_probs)
        predicted_probs = exp_probs / np.sum(exp_probs)

        predicted_idx = np.random.choice(len(predicted_probs), p=predicted_probs)
        generated.append(int_to_note[predicted_idx])
    return generated


# 5. MIDI文件生成
def save_midi(generated_notes, filename="output.mid"):
    midi_stream = stream.Stream()
    for n in generated_notes:
        if '.' in n:  # 处理和弦
            chord_notes = chord.Chord([int(p) for p in n.split('.')])
            midi_stream.append(chord_notes)
        else:  # 处理单音符
            midi_note = note.Note(int(n))
            midi_stream.append(midi_note)
    midi_stream.write('midi', fp=filename)


# 主流程
if __name__ == "__main__":
    # 参数设置
    sequence_length = 100
    epochs = 150
    batch_size = 64

    # 数据加载与处理
    notes = load_midi_data("input.mid")
    X, y, note_to_int = create_sequences(notes, sequence_length)
    int_to_note = {v: k for k, v in note_to_int.items()}

    # 模型训练
    model = build_model(X.shape[1], len(note_to_int))
    model.fit(X, y, epochs=epochs, batch_size=batch_size)

    # 生成新音乐
    start_sequence = notes[:sequence_length]
    generated_notes = generate_music(model, start_sequence, note_to_int, int_to_note)
    save_midi(generated_notes)