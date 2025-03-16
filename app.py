from flask import Flask, request, render_template, jsonify
import torch
import librosa
import numpy as np
import pyaudio
import soundfile as sf
import matplotlib.pyplot as plt

# Initialize Flask app
app = Flask(__name__)

# Track emotions over time
emotion_history = []

# Define emotions list
emotions = ["neutral", "happy", "sad", "angry", "fearful", "disgust", "surprised"]

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
    FORMAT = pyaudio.paInt16  # Ensure it's 16-bit PCM
    CHANNELS = 1

    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=sample_rate, input=True,
                        frames_per_buffer=CHUNK)

    print("🎙️ Recording... Speak now!")
    frames = []

    # Read audio data in chunks
    for _ in range(0, int(sample_rate / CHUNK * duration)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("✅ Recording complete!")

    stream.stop_stream()
    stream.close()
    audio.terminate()

    # Convert byte data to int16 array
    audio_data = np.frombuffer(b''.join(frames), dtype=np.int16)

    # Save as WAV file
    sf.write(file_path, audio_data, sample_rate, subtype='PCM_16')


# -------------------------------
# ROUTES
# -------------------------------

# Home Route
@app.route("/")
def home():
    return render_template("index.html")


# Route to Predict Emotion from Uploaded File
@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["file"]
    file_path = "uploaded_audio.wav"
    file.save(file_path)

    # Extract features and make a prediction
    features = torch.tensor(extract_features_live(file_path), dtype=torch.float32)
    prediction = model(features)
    emotion = torch.argmax(prediction, axis=1).item()

    result = emotions[emotion]

    # Track the emotion
    emotion_history.append(result)

    return f"Predicted Emotion: {result}"


# Route to Record Audio and Predict Emotion
@app.route("/record", methods=["POST"])
def record_and_predict():
    file_path = "live_recording.wav"
    record_audio(file_path)

    # Extract features and make a prediction
    features = torch.tensor(extract_features_live(file_path), dtype=torch.float32)
    prediction = model(features)
    emotion = torch.argmax(prediction, axis=1).item()

    result = emotions[emotion]

    # Track the emotion
    emotion_history.append(result)

    return f"Predicted Emotion (Live): {result}"


# Route to Generate and Display Emotion Tracking Graph
@app.route("/emotion_graph")
def emotion_graph():
    if not emotion_history:
        return "No emotions tracked yet."

    plt.figure(figsize=(10, 5))
    plt.plot(range(len(emotion_history)), emotion_history, color='blue', marker='o', linewidth=2)
    plt.title('Emotion Tracking Over Time')
    plt.xlabel('Recording Number')
    plt.ylabel('Detected Emotion')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()

    # Save the graph as an image
    plt.savefig("static/emotion_graph.png")
    plt.close()

    return render_template("graph.html", img_url="/static/emotion_graph.png")


# -------------------------------
# RUN FLASK APP
# -------------------------------
if __name__ == "__main__":
    app.run(debug=True)
