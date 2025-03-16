from flask import Flask, request, render_template, jsonify
import torch
import torch.nn as nn
import numpy as np
import librosa
import matplotlib.pyplot as plt
import pyaudio
import soundfile as sf


# -------------------------------
# DEFINE THE MODEL ARCHITECTURE
# -------------------------------
class SpeechEmotionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SpeechEmotionModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.fc(x)


# -------------------------------
# LOAD THE MODEL
# -------------------------------
input_dim = 40  # Number of MFCC features
output_dim = 7  # Number of emotion classes
model = SpeechEmotionModel(input_dim, output_dim)

# Load the saved model
try:
    model.load_state_dict(torch.load("speech_emotion_model.pth", map_location=torch.device("cpu")))
    model.eval()
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")


# -------------------------------
# FLASK APP SETUP
# -------------------------------
app = Flask(__name__)


# -------------------------------
# FEATURE EXTRACTION FUNCTION
# -------------------------------
def extract_features_live(file_path):
    y, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    return np.mean(mfcc.T, axis=0)

# Track emotions across multiple predictions
emotion_history = []

# -------------------------------
# Microphone Recording Function
# -------------------------------
def record_audio(file_path, duration=5, sample_rate=22050):
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1

    # Initialize PyAudio
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=sample_rate, input=True,
                        frames_per_buffer=CHUNK)

    print("🎙️ Recording... Speak now!")
    frames = []

    # Capture audio data in chunks
    for _ in range(0, int(sample_rate / CHUNK * duration)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("✅ Recording complete!")

    stream.stop_stream()
    stream.close()
    audio.terminate()

    # Convert byte data to int16 array
    audio_data = np.frombuffer(b''.join(frames), dtype=np.int16)

    # Save audio file
    sf.write(file_path, audio_data, sample_rate, subtype='PCM_16')

# -------------------------------
# HOME ROUTE
# -------------------------------
@app.route("/")
def home():
    return render_template("index.html")


# -------------------------------
# PREDICTION ROUTE
# -------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["file"]
    file_path = "uploaded_audio.wav"
    file.save(file_path)

    features = torch.tensor(extract_features_live(file_path), dtype=torch.float32).unsqueeze(0)

    prediction = model(features)
    emotion = torch.argmax(prediction, axis=1).item()

    emotions = ["neutral", "happy", "sad", "angry", "fearful", "disgust", "surprised"]
    result = emotions[emotion]

    # Track this emotion
    emotion_history.append(result)

    return f"Predicted Emotion: {result}"

#--------------------------------
# RECORD
#--------------------------------
@app.route("/record", methods=["POST"])
def record_and_predict():
    file_path = "live_recording.wav"
    record_audio(file_path)

    features = torch.tensor(extract_features_live(file_path), dtype=torch.float32).unsqueeze(0)
    prediction = model(features)
    emotion = torch.argmax(prediction, axis=1).item()

    emotions = ["neutral", "happy", "sad", "angry", "fearful", "disgust", "surprised"]
    result = emotions[emotion]

    # Track this emotion
    emotion_history.append(result)

    return f"Predicted Emotion (Live): {result}"

#--------------------------------
# GRAPH
#--------------------------------
@app.route("/emotion_graph")
def emotion_graph():
    if not emotion_history:
        return "No emotions tracked yet!"

    plt.figure(figsize=(10, 5))
    plt.plot(range(len(emotion_history)), emotion_history, color='blue', marker='o', linewidth=2)
    plt.title('Emotion Tracking Over Time')
    plt.xlabel('Recording Number')
    plt.ylabel('Detected Emotion')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()

    plt.savefig("static/emotion_graph.png")
    plt.close()

    return render_template("graph.html", img_url="/static/emotion_graph.png")

# -------------------------------
# RUN FLASK APP
# -------------------------------
if __name__ == "__main__":
    app.run(debug=True)
