import os
import librosa
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Feature Extraction Function
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    return np.mean(mfcc.T, axis=0)

# Load Dataset
DATASET_PATH = "path_to_dataset"
X, Y = [], []

for folder in os.listdir(DATASET_PATH):
    for file in os.listdir(os.path.join(DATASET_PATH, folder)):
        file_path = os.path.join(DATASET_PATH, folder, file)
        feature_vector = extract_features(file_path)
        X.append(feature_vector)
        Y.append(folder)

# Encode Labels
label_encoder = LabelEncoder()
Y = label_encoder.fit_transform(Y)

# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    np.array(X), np.array(Y), test_size=0.2, random_state=42
)

print("✅ Data Loaded Successfully!")


import torch
import torch.nn as nn
import torch.optim as optim

# Define Model
class SpeechEmotionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SpeechEmotionModel, self).__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(input_dim, 128, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(128 * 2, output_dim)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension
        x = torch.relu(self.conv1(x))
        _, (h, _) = self.lstm(x)
        x = torch.cat((h[0], h[1]), dim=1)
        x = self.dropout(x)
        return self.fc(x)


# Model Setup
input_dim = X_train.shape[1]
output_dim = len(np.unique(Y))
model = SpeechEmotionModel(input_dim=input_dim, output_dim=output_dim)

# Define Loss & Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("✅ Model Initialized Successfully!")


# Convert data to tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)

X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Training Loop
num_epochs = 30
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()

    if (epoch+1) % 5 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

print("✅ Model Trained Successfully!")


from sklearn.metrics import classification_report, accuracy_score

model.eval()
with torch.no_grad():
    y_pred = model(X_test_tensor)
    y_pred_classes = torch.argmax(y_pred, axis=1)

# Report Accuracy
print("Accuracy:", accuracy_score(y_test_tensor, y_pred_classes))
print("\nClassification Report:\n", classification_report(y_test_tensor, y_pred_classes))


torch.save(model.state_dict(), "speech_emotion_model.pth")
print("✅ Model Saved Successfully!")
