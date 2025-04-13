import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import sounddevice as sd
import numpy as np
import librosa
from tensorflow.keras.models import load_model
import time

# Load your trained model
model = load_model("Chicken_Audio.keras")

# Define your class names
CATEGORIES = ["Healthy", "Unhealthy", "Noise"] # <- Update with your real class labels

# Parameters
sr = 22050          # Sample rate
duration = 2        # seconds
threshold = 0.6     # Confidence threshold (optional)

# Feature extraction
def extract_features_from_audio(y, sr=22050, n_mels=128, n_fft=2048, hop_length=512):
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
    log_mel_spec = librosa.power_to_db(mel_spec)
    features = np.mean(log_mel_spec, axis=1)  # shape: (128,)
    return features

# Start continuous prediction
print("🎤 Listening for chicken sounds... (Press Ctrl+C to stop)")
try:
    while True:
        audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32')
        sd.wait()

        y = audio.flatten()
        if np.abs(y).mean() < 0.01:
            print("🔇 Silent / background noise")
            continue

        features = extract_features_from_audio(y, sr=sr)
        features = features.reshape(features.shape[0], 1)   # (128, 1)
        features = np.expand_dims(features, axis=0)         # (1, 128, 1)

        prediction = model.predict(features, verbose=0)
        confidence = np.max(prediction)
        predicted_index = np.argmax(prediction)

        if confidence >= threshold:
            label = CATEGORIES[predicted_index]
            print(f"✅ Detected: {label} (Confidence: {confidence:.2f})")
        else:
            print("🤔 Low confidence prediction")

        time.sleep(0.25)

except KeyboardInterrupt:
    print("🛑 Stopped listening.")
