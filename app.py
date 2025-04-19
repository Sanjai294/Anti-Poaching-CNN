import sounddevice as sd
import numpy as np
import librosa
from tensorflow.keras.models import load_model

# Preprocessing function
def preprocess_audio(audio, sr=22050, n_mfcc=13, hop_length=512, n_fft=2048):
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length, n_fft=n_fft, res_type='kaiser_fast')
    max_length = 100  # Adjust to your model’s input
    if mfccs.shape[0] > max_length:
        mfccs = mfccs[:max_length]
    else:
        mfccs = np.pad(mfccs, ((0, max_length - mfccs.shape[0]), (0, 0)), mode='constant')
    return mfccs[np.newaxis, ..., np.newaxis]

# Load model
model = load_model('/home/sanjai/mlprojects/Unisys/saved_model/my_model.keras')
class_labels = ['class1', 'class2', 'class3']  # Your classes
sample_rate = 22050
chunk_duration = 1.0
chunk_samples = int(sample_rate * chunk_duration)

# Audio callback
def audio_callback(indata, frames, time, status):
    if status:
        print(f"Error: {status}")
        return
    try:
        audio_chunk = indata[:, 0]
        features = preprocess_audio(audio_chunk, sr=sample_rate)
        prediction = model.predict(features, verbose=0)
        pred_class = class_labels[np.argmax(prediction)]
        confidence = prediction[0][np.argmax(prediction)]
        print(f"Predicted: {pred_class} (Confidence: {confidence:.2f})")
    except Exception as e:
        print(f"Processing error: {e}")

# Start stream
print("Listening... Press Ctrl+C to stop.")
with sd.InputStream(samplerate=sample_rate, channels=1, blocksize=chunk_samples, callback=audio_callback):
    try:
        while True:
            sd.sleep(100)
    except KeyboardInterrupt:
        print("Stopped.")