import os
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

base_dir = "/home/sanjai/mlprojects/Unisys/data"
metadata_file = os.path.join(base_dir, "UrbanSound8K.csv")
n_mels = 128
n_fft = 2048
hop_length = 512
time_frames = 128

audiofiles = pd.read_csv(metadata_file)
print(f"Total entries: {len(audiofiles)}")

# EXTRACT MEL SPECTROGRAM
def extract_mel(file_path, n_mels, n_fft, hop_length, time_frames):
    y, sr = librosa.load(file_path, sr=None)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    
    # Pad or trim
    if mel_db.shape[1] > time_frames:
        mel_db = mel_db[:, :time_frames]
    elif mel_db.shape[1] < time_frames:
        pad_width = time_frames - mel_db.shape[1]
        mel_db = np.pad(mel_db, ((0, 0), (0, pad_width)), mode='constant')
    
    return mel_db

# PROCESS ALL FILES
X_data = []
y_data = []

for i in range(len(audiofiles)):
    if i % 100 == 0:
        print(f"Processing {i}/{len(audiofiles)}")

    row = audiofiles.iloc[i]
    file_path = os.path.join(base_dir, f"fold{row['fold']}", row['slice_file_name'])

    if not os.path.exists(file_path):
        print(f"Missing: {file_path}")
        continue

    try:
        mel = extract_mel(file_path, n_mels, n_fft, hop_length, time_frames)
        X_data.append(mel)
        y_data.append(row['classID'])
    except Exception as e:
        print(f"Error in {file_path}: {e}")
        continue
    
    X_data = np.array(X_data)
y_data = np.array(y_data)
print(f"X_data shape before reshape: {X_data.shape}")

X_data = X_data[..., np.newaxis]  # (samples, n_mels, time_frames, 1)
X_data = (X_data - np.mean(X_data)) / np.std(X_data)
print("Unique classIDs in dataset:", np.unique(y_data))
print("Total number of unique classes:", len(np.unique(y_data)))

# Remap class IDs to contiguous labels
unique_classes = sorted(np.unique(y_data))
print("Original Class IDs:", unique_classes)

label_map = {orig: idx for idx, orig in enumerate(unique_classes)}
inverse_label_map = {v: k for k, v in label_map.items()}

# Apply remapping
y_data_mapped = np.array([label_map[label] for label in y_data])

# One-hot encode using the remapped labels
y_categorical = to_categorical(y_data_mapped)

print("Remapped class IDs:", np.unique(y_data_mapped))
print(f"Final input shape: {X_data.shape}, Labels: {y_categorical.shape}")

# Encode Labels
y_categorical = to_categorical(y_data)
print(f"Final input shape: {X_data.shape}, Labels: {y_categorical.shape}")

X_train, X_test, y_train, y_test = train_test_split(X_data, y_categorical, test_size=0.2, random_state=42, stratify=y_data)

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=X_train.shape[1:]),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.2),

    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.3),

    Conv2D(128, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.4),

    Flatten(),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dense(y_categorical.shape[1], activation='softmax')
])

model.summary()

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ModelCheckpoint('best_audio_cnn_model.keras', monitor='val_accuracy', save_best_only=True)
]

history = model.fit(X_train, y_train,validation_data=(X_test, y_test),epochs=50,batch_size=32,callbacks=callbacks,verbose=1)


test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc:.4f}, Test Loss: {test_loss:.4f}")