import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# -------------------------------
# DATA LOADING & FEATURE EXTRACTION
# -------------------------------
def extract_features(file_path):
    """Extracts MFCC features from an audio file."""
    y, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    return np.mean(mfcc.T, axis=0)

# Load dataset
DATASET_PATH = "data"
X, Y = [], []

# Load data from each folder (one folder per emotion)
for folder in os.listdir(DATASET_PATH):
    folder_path = os.path.join(DATASET_PATH, folder)
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        features = extract_features(file_path)
        X.append(features)
        Y.append(folder)

# Encode labels (convert emotions to numbers)
encoder = LabelEncoder()
Y = encoder.fit_transform(Y)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(np.array(X), np.array(Y), test_size=0.2, random_state=42)

# -------------------------------
# DEFINE THE MODEL
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

# Create model
input_dim = X_train.shape[1]
output_dim = len(set(Y))
model = SpeechEmotionModel(input_dim, output_dim)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# -------------------------------
# TRAIN THE MODEL
# -------------------------------
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    # Convert data to tensors
    inputs = torch.tensor(X_train, dtype=torch.float32)
    labels = torch.tensor(y_train, dtype=torch.long)

    # Forward pass
    outputs = model(inputs)
    loss = criterion(outputs, labels)

    # Backward pass and optimize
    loss.backward()
    optimizer.step()

    # Print progress every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# -------------------------------
# EVALUATE THE MODEL
# -------------------------------
model.eval()
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

with torch.no_grad():
    predictions = model(X_test_tensor)
    predicted_classes = torch.argmax(predictions, axis=1)
    accuracy = (predicted_classes == y_test_tensor).sum().item() / len(y_test_tensor)

print(f"✅ Model Accuracy: {accuracy:.2f}")

# -------------------------------
# SAVE THE MODEL
# -------------------------------
torch.save(model.state_dict(), "speech_emotion_model.pth")
print("🎉 Model saved as 'speech_emotion_model.pth'")
