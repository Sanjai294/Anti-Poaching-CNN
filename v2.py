import librosa
import tensorflow as tf
import sys
import os

# === Configuration ===
model_path = '/home/sanjai/mlprojects/Unisys/saved_model/my_model_v2.keras'
audio_path = '/home/sanjai/mlprojects/Unisys/data/fold11/weapon_gun_rattle_out_32.wav'  # Update this path as needed
class_labels = ['air_conditioner', 'car_horn', 'children_playing', 'dog_bark', 'drilling',
                'engine_idling', 'gun_shot', 'jackhammer', 'siren', 'street_music']

# === Load the model ===
model = tf.keras.models.load_model(model_path)
print("✅ Model loaded from:", model_path)

# === Load audio ===
y, sr = librosa.load(audio_path, sr=22050, mono=True)
print("🎧 Audio loaded from:", audio_path)

# === Extract log-mel spectrogram ===
n_mels = 128
hop_length = 512
n_fft = 2048

mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft,
                                          hop_length=hop_length, n_mels=n_mels)
log_mel_spec = librosa.power_to_db(mel_spec)

# === Resize or pad to fixed shape (128, 128) ===
target_shape = (128, 128)
if log_mel_spec.shape[1] < target_shape[1]:
    pad_width = target_shape[1] - log_mel_spec.shape[1]
    log_mel_spec = np.pad(log_mel_spec, ((0, 0), (0, pad_width)), mode='constant')
else:
    log_mel_spec = log_mel_spec[:, :target_shape[1]]

# === Normalize and reshape ===
log_mel_spec = (log_mel_spec - np.mean(log_mel_spec)) / (np.std(log_mel_spec) + 1e-6)
input_data = np.expand_dims(log_mel_spec, axis=-1)  # (128, 128, 1)
input_data = np.expand_dims(input_data, axis=0)     # (1, 128, 128, 1)

# === Predict ===
pred = model.predict(input_data)
predicted_index = np.argmax(pred)
predicted_label = class_labels[predicted_index]
confidence = pred[0][predicted_index]

# === Output ===
print("\n🔮 Prediction vector:", np.round(pred, 3))
print("✅ Predicted class index:", predicted_index)
print("🎧 Predicted label:", predicted_label)
print("📊 Confidence:", round(confidence, 2))
