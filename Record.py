import numpy as np
from music21 import converter, instrument, note, chord, stream
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Embedding, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import LearningRateScheduler
import pretty_midi
import librosa
import sounddevice as sd


# 1. 音频录制
def record_audio(duration=5, fs=44100):
    print("开始录制...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    print("录制结束")
    return audio.flatten()


# 2. 音频转音高序列
def audio_to_pitch_sequence(audio, fs=44100):
    pitches, magnitudes = librosa.piptrack(y=audio, sr=fs)
    pitch_sequence = []
    for i in range(pitches.shape[1]):
        index = magnitudes[:, i].argmax()
        pitch = pitches[index, i]
        if pitch > 0:
            midi_note = librosa.hz_to_midi(pitch)
            pitch_sequence.append(str(int(midi_note)))
    return pitch_sequence


# 3. 序列编码与数据集构建
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


# 4. LSTM模型构建
def build_model(input_shape, num_unique_notes):
    model = Sequential([
        Embedding(input_dim=num_unique_notes, output_dim=64),
        LSTM(256, return_sequences=True),
        BatchNormalization(),
        Dropout(0.3),
        LSTM(256),
        BatchNormalization(),
        Dense(num_unique_notes, activation='softmax')
    ])
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')
    return model


# 学习率调度器
def lr_scheduler(epoch, lr):
    if epoch % 20 == 0 and epoch > 0:
        lr = lr * 0.9
    return lr


# 5. 和声生成
def generate_harmony(model, start_sequence, note_to_int, int_to_note, length=500, temperature=0.7):
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


# 6. 配器
def add_instruments(midi_stream):
    # 为旋律添加钢琴
    melody_part = stream.Part()
    melody_part.insert(0, instrument.Piano())
    for element in midi_stream:
        melody_part.append(element)

        # 为和声添加弦乐
    harmony_part = stream.Part()
    harmony_part.insert(0, instrument.StringInstrument())

    # 这里简单示例，和声与旋律相同，可根据实际情况修改
    for element in midi_stream:
        harmony_part.append(element)

    final_stream = stream.Score()
    final_stream.insert(0, melody_part)
    final_stream.insert(0, harmony_part)
    return final_stream


# 7. MIDI文件生成
def save_midi(generated_notes, filename="output.mid"):
    midi_stream = stream.Stream()
    for n in generated_notes:
        if '.' in n:  # 处理和弦
            chord_notes = chord.Chord([int(p) for p in n.split('.')])
            midi_stream.append(chord_notes)
        else:  # 处理单音符
            midi_note = note.Note(int(n))
            midi_stream.append(midi_note)
    midi_stream = add_instruments(midi_stream)
    midi_stream.write('midi', fp=filename)


# 主流程
if __name__ == "__main__":
    # 参数设置
    sequence_length = 100
    epochs = 200
    batch_size = 64

    # 音频录制与处理
    audio = record_audio()
    pitch_sequence = audio_to_pitch_sequence(audio)

    # 数据处理
    X, y, note_to_int = create_sequences(pitch_sequence, sequence_length)
    int_to_note = {v: k for k, v in note_to_int.items()}

    # 模型训练
    model = build_model(X.shape[1], len(note_to_int))
    lr_callback = LearningRateScheduler(lr_scheduler)
    model.fit(X, y, epochs=epochs, batch_size=batch_size, callbacks=[lr_callback])

    # 生成和声
    start_sequence = pitch_sequence[:sequence_length]
    generated_harmony = generate_harmony(model, start_sequence, note_to_int, int_to_note)

    # 生成新音乐
    generated_notes = pitch_sequence + generated_harmony
    save_midi(generated_notes)