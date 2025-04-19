import numpy as np
import librosa
import tensorflow as tf

# === Path to your saved model (.keras) ===
model_path = '/home/sanjai/mlprojects/Unisys/saved_model/CNN_V2.keras'

# ✅ Corrected path to shared audio file (Linux-style)
audio_path = '/mnt/c/Users/ADMIN/Desktop/UnisysAudio/output.wav'  # <- This is the correct WSL path

# === Class labels used during model training ===
class_labels = ['air_conditioner', 'car_horn', 'children_playing', 'dog_bark', 'drilling',
                'engine_idling', 'gun_shot', 'jackhammer', 'siren', 'street_music']

# === Load model ===
model = tf.keras.models.load_model(model_path)

# === Load audio ===
y, sr = librosa.load(audio_path, sr=22050)  # Resample if needed

# === Extract MFCCs ===
mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=128)
mfcc_mean = np.mean(mfcc.T, axis=0)  # shape: (128,)

# === Prepare input for model ===
input_data = np.expand_dims(mfcc_mean, axis=0)  # shape: (1, 128)

# === Make prediction ===
prediction = model.predict(input_data)
predicted_index = np.argmax(prediction)
predicted_label = class_labels[predicted_index]

# === Output ===
print("🔮 Prediction vector:", prediction)
print("✅ Predicted class index:", predicted_index)
print("🎧 Predicted label:", predicted_label)
