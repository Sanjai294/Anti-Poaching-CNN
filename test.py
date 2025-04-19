import numpy as np
import librosa
import sounddevice as sd
from tensorflow.keras.models import load_model
import argparse
import os
from IPython.display import Audio
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Define class labels (UrbanSound8K classes)
class_labels = ['dog_bark', 'children_playing', 'car_horn', 'air_conditioner',
       'street_music', 'gun_shot', 'siren', 'engine_idling', 'jackhammer',
       'drilling', 'chainsaw', 'footsteps', 'radio_human_sound',
       'weapon_clink']

# Preprocessing function (matches training)
def preprocess_audio(audio, sr=22050, n_mels=128, hop_length=512, n_fft=2048):
    mels = np.mean(librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels, hop_length=hop_length, n_fft=n_fft).T, axis=0)
    return mels

# File-based prediction
def predict_file(filename, model, sample_rate=22050):
    if not os.path.exists(filename):
        print(f"Error: File {filename} not found.")
        return None
    try:
        audio, sr = librosa.load(filename, sr=sample_rate, res_type='kaiser_fast')
        features = preprocess_audio(audio, sr=sample_rate)
        X = np.array([features])
        prediction = model.predict(X, verbose=0)
        class_id = np.argmax(prediction)
        print(f"File: {filename}")
        print(f"Predicted class: {class_labels[class_id]} (Confidence: {prediction[0][class_id]:.2f})")
        return Audio(data=audio, rate=sr)
    except Exception as e:
        print(f"Error processing {filename}: {e}")
        return None

# Live audio prediction
def predict_live(model, sample_rate=22050, chunk_duration=1.0):
    chunk_samples = int(sample_rate * chunk_duration)
    
    def audio_callback(indata, frames, time, status):
        if status:
            print(f"Stream status: {status}")
        audio_chunk = indata[:, 0]
        features = preprocess_audio(audio_chunk, sr=sample_rate)
        X = np.array([features])
        prediction = model.predict(X, verbose=0)
        class_id = np.argmax(prediction)
        print(f"Predicted: {class_labels[class_id]} (Confidence: {prediction[0][class_id]:.2f})")

    print("Listening... Press Ctrl+C to stop.")
    try:
        with sd.InputStream(samplerate=sample_rate, channels=1, blocksize=chunk_samples, callback=audio_callback):
            while True:
                pass
    except sd.PortAudioError as e:
        print(f"Audio device error: {e}. Ensure a microphone is connected and PulseAudio is configured.")
    except KeyboardInterrupt:
        print("Stopped.")

def main():
    parser = argparse.ArgumentParser(description="UrbanSound8K Audio Classifier")
    parser.add_argument('--model', type=str, default='/home/sanjai/mlprojects/Unisys/saved_model/my_model.keras',
                        help='Path to the trained Keras model')
    parser.add_argument('--mode', type=str, choices=['file', 'live'], default='file',
                        help='Prediction mode: "file" or "live"')
    parser.add_argument('--input', type=str, default=None,
                        help='Input audio file for file mode')
    args = parser.parse_args()

    # Load model
    try:
        model = load_model(args.model)
        print(f"Loaded model from {args.model}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Run prediction
    if args.mode == 'file':
        if not args.input:
            print("Error: --input required for file mode.")
            return
        predict_file(args.input, model)
    else:
        predict_live(model)

if __name__ == "__main__":
    main()