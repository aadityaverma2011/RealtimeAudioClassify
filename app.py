import streamlit as st
import numpy as np
import librosa
import torch
import torch.nn as nn
import tempfile

from audio_recorder_streamlit import audio_recorder

# === Your CNN model ===
class AudioCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(AudioCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(64 * 8 * 16, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = self.dropout(torch.relu(self.fc1(x)))
        return self.fc2(x)

# === Load the trained model ===
@st.cache_resource
def load_model():
    model = AudioCNN()
    model.load_state_dict(torch.load("audio_cnn.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

# === Label classes ===
CLASS_LABELS = [
    "Air Conditioner", "Car Horn", "Children Playing", "Dog Bark", "Drilling",
    "Engine Idling", "Gun Shot", "Jackhammer", "Siren", "Street Music"
]

# === Mel spectrogram extractor ===
def extract_mel(audio, sr, max_len=128):
    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=64)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    if mel_db.shape[1] < max_len:
        pad = max_len - mel_db.shape[1]
        mel_db = np.pad(mel_db, ((0, 0), (0, pad)), mode='constant')
    else:
        mel_db = mel_db[:, :max_len]
    return mel_db

# === Streamlit UI ===
st.title("Urban Sound Classifier")
st.markdown("Record audio and classify it using a CNN trained on UrbanSound8K.")
with st.expander("What can this app detect?"):
    st.markdown("""
This app can classify **urban environmental sounds** based on a deep learning model trained on the **UrbanSound8K** dataset.

**It can detect the following 10 sound classes only**:

- Car Horn  
- Children Playing  
- Dog Bark  
- Drilling  
- Engine Idling  
- Gun Shot  
- Jackhammer  
- Siren  
- Street Music  
- Air Conditioner

---

**Important Notes**:
- The app will **not** recognize sounds outside of these categories.
- Accuracy depends on background noise and microphone quality.
- For best results, use clear, isolated samples of one sound class.
    """)

# === Record audio ===
audio_bytes = audio_recorder(pause_threshold=3.0, energy_threshold=0.003)

if audio_bytes:
    st.audio(audio_bytes, format="audio/wav")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
        tmpfile.write(audio_bytes)
        tmpfile.flush()

        if st.button("Predict"):
            audio, sr = librosa.load(tmpfile.name, sr=None)
            mel = extract_mel(audio, sr)
            mel_tensor = torch.tensor(mel).unsqueeze(0).unsqueeze(0).float()
            model = load_model()
            with torch.no_grad():
                output = model(mel_tensor)
                predicted = torch.argmax(output, dim=1).item()
                label = CLASS_LABELS[predicted]
                st.success(f"Predicted Sound Class: **{label}**")
