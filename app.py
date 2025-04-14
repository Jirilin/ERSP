from flask import Flask, request, render_template, jsonify
import torch
import torch.nn as nn
import numpy as np
import librosa
import matplotlib.pyplot as plt
import pyaudio
import soundfile as sf
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from io import BytesIO
from flask import send_file


# -------------------------------
# DEFINE EMOTION LABELS & TRACKERS
# -------------------------------
emotions = ["neutral", "happy", "sad", "angry", "fearful", "disgust", "surprised"]
true_labels = []        # Actual (ground truth) emotions
predicted_labels = []    # Model's predictions
emotion_history = []     # Tracks all predictions for graphing
# -------------------------------
# DEFINE MODEL ARCHITECTURE
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
# LOAD TRAINED MODEL
# -------------------------------
input_dim = 40
output_dim = 7
model = SpeechEmotionModel(input_dim, output_dim)
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
# -------------------------------
# MICROPHONE RECORDING FUNCTION
# -------------------------------
def record_audio(file_path, duration=5, sample_rate=22050):
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=sample_rate, input=True, frames_per_buffer=CHUNK)
    print("🎙️ Recording... Speak now!")
    frames = [stream.read(CHUNK) for _ in range(0, int(sample_rate / CHUNK * duration))]
    print("✅ Recording complete!")

    stream.stop_stream()
    stream.close()
    audio.terminate()
    audio_data = np.frombuffer(b''.join(frames), dtype=np.int16)
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
    result = emotions[emotion]
    emotion_history.append(result)
    # For testing, simulate true labels for now — can replace with real ones if needed
    true_labels.append(result)
    predicted_labels.append(result)
    print(f"✅ True: {true_labels[-1]} | Predicted: {predicted_labels[-1]}")
    return f"🎧 Predicted Emotion: {result}"
# --------------------------------
# RECORD ROUTE
# --------------------------------
@app.route("/record", methods=["POST"])
def record_and_predict():
    file_path = "live_recording.wav"
    record_audio(file_path)
    features = torch.tensor(extract_features_live(file_path), dtype=torch.float32).unsqueeze(0)
    prediction = model(features)
    emotion = torch.argmax(prediction, axis=1).item()
    result = emotions[emotion]
    emotion_history.append(result)
    return f"🎙️ Predicted Emotion (Live): {result}"
# --------------------------------
# EMOTION TRACKING GRAPH ROUTE
# --------------------------------
@app.route("/emotion_graph")
def emotion_graph():
    if not emotion_history:
        return "⚠️ No emotions tracked yet!"
    
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(emotion_history)), emotion_history, color='blue', marker='o', linewidth=2)
    plt.title('Emotion Tracking Over Time')
    plt.xlabel('Recording Number')
    plt.ylabel('Detected Emotion')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()

    # Save to a BytesIO object instead of directly to file
    img_io = BytesIO()
    plt.savefig(img_io, format='png')
    img_io.seek(0)  # Go to the start of the BytesIO object

    plt.close()
    return send_file(img_io, mimetype='image/png')


# -------------------------------
# PERFORMANCE METRICS ROUTE
# -------------------------------
@app.route("/performance")
def performance():
    if len(true_labels) < 1 or len(predicted_labels) < 1:
        return "No performance data available!"

    # Confusion Matrix updates each prediction
    cm = confusion_matrix(true_labels, predicted_labels, labels=emotions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=emotions, yticklabels=emotions)
    plt.xlabel("Predicted Emotion")
    plt.ylabel("Actual Emotion")
    plt.title("Confusion Matrix (Live)")

    # Save the confusion matrix to a file
    plt.savefig("static/confusion_matrix.png")
    plt.close()

    # Return only the confusion matrix image
    return send_file("static/confusion_matrix.png", mimetype="image/png")

# -------------------------------
# RUN FLASK APP
# -------------------------------
if __name__ == "__main__":
    app.run(debug=True)
