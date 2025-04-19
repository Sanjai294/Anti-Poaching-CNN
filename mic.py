import sounddevice as sd
from scipy.io.wavfile import write

fs = 44100  # Sample rate
duration = 5  # Duration in seconds

print("Recording...")
recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
sd.wait()  # Wait until recording is finished
write("output.wav", fs, recording)
print("Saved as output.wav")

print(sd.query_devices())
