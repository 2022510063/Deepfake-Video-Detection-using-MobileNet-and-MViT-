import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import streamlit as st
from torchvision import models
from torchvision.models.video import mvit_v2_s
import tempfile

# Set up device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
st.write(f"Using device: {device}")

# Define MViT model
class MViTVideoClassifier(nn.Module):
    def __init__(self, num_classes=1, dropout=0.4):
        super().__init__()
        self.backbone = mvit_v2_s(weights=None)
        features = self.backbone.head[1].in_features
        self.backbone.head = nn.Sequential(
            nn.Linear(features, features // 2),
            nn.LayerNorm(features // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(features // 2, num_classes),
            nn.Sigmoid() if num_classes == 1 else nn.Identity()
        )

    def forward(self, x):
        x_permuted = x.permute(0, 2, 1, 3, 4)
        x = self.backbone(x_permuted)
        outputs = x.squeeze()
        if isinstance(outputs, torch.Tensor) and outputs.dim() == 0:
            outputs = outputs.unsqueeze(0)
        return outputs

# Define MobileNetV2 model
class MobileNetV2Classifier(nn.Module):
    def __init__(self, num_classes=1, dropout=0.4):
        super().__init__()
        self.backbone = models.mobilenet_v2(weights=None)
        features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Linear(features, features // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(features // 2, num_classes),
            nn.Sigmoid() if num_classes == 1 else nn.Identity()
        )

    def forward(self, x):
        batch, seq_len, channels, height, width = x.size()
        x = x.view(batch * seq_len, channels, height, width)
        outputs = self.backbone(x)
        outputs = outputs.view(batch, seq_len, -1)
        outputs = outputs.mean(dim=1).squeeze()
        if isinstance(outputs, torch.Tensor) and outputs.dim() == 0:
            outputs = outputs.unsqueeze(0)
        return outputs

# Load model based on user selection
@st.cache_resource
def load_model(model_type, model_path):
    if model_type == "MViT":
        model = MViTVideoClassifier(num_classes=1, dropout=0.4).to(device)
    else:  # MobileNetV2
        model = MobileNetV2Classifier(num_classes=1, dropout=0.4).to(device)
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        st.warning(f"Model file {model_path} not found. Using untrained model.")
    model.eval()
    return model

# Image preprocessing
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Function to extract frames from video
def extract_frames(video_path, num_frames=10, clip_duration=2.0):
    frames = []
    vidcap = cv2.VideoCapture(video_path)
    
    if not vidcap.isOpened():
        return None
    
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0 or fps <= 0:
        return None
    
    frames_needed = min(num_frames, total_frames)
    frame_interval = max(1, int((clip_duration * fps) / frames_needed))
    frame_indices = list(range(0, min(total_frames, int(clip_duration * fps)), frame_interval))[:frames_needed]
    
    for frame_idx in frame_indices:
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        success, frame = vidcap.read()
        if success:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        else:
            frames.append(np.zeros_like(frames[-1]) if frames else np.zeros((224, 224, 3), dtype=np.uint8))
    
    vidcap.release()
    return np.stack(frames)

# Function to preprocess frames
def preprocess_frames(frames):
    processed_frames = [transform(frame) for frame in frames]
    video_tensor = torch.stack(processed_frames, dim=0).unsqueeze(0).to(device)
    return video_tensor

# Function to predict if a video is real or fake
def predict_video(video_path, model):
    frames = extract_frames(video_path, num_frames=16)  # changed to 16
    
    if frames is None or len(frames) == 0:
        return {"error": "Could not extract frames from video"}
    
    video_tensor = preprocess_frames(frames)
    
    with torch.no_grad():
        outputs = model(video_tensor)
        fake_probability = outputs.item()
    
    is_fake = fake_probability > 0.5
    
    return {
        "is_fake": bool(is_fake),
        "confidence": float(fake_probability if is_fake else 1 - fake_probability),
        "frame_count": len(frames)
    }


# Streamlit UI
st.title("Deepfake Video Detection")
st.write("Upload a video and choose a model to detect if it's real or fake.")

# Model selection
model_type = st.selectbox("Select Model", ["MViT", "MobileNetV2"])
model_path = f"models/{'mvit' if model_type == 'MViT' else 'mobilenetv2'}_best_model.pth"
model = load_model(model_type, model_path)

# File uploader
uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    # Save the uploaded video to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        tmp_file.write(uploaded_file.read())
        video_path = tmp_file.name

    # Display the uploaded video
    st.video(video_path)

    # Predict button
    if st.button("Analyze Video"):
        with st.spinner("Analyzing video..."):
            result = predict_video(video_path, model)
        
        # Display results
        if "error" in result:
            st.error(result["error"])
        else:
            status = "Fake" if result["is_fake"] else "Real"
            confidence = result["confidence"]
            frame_count = result["frame_count"]
            
            st.write(f"**Prediction**: {status}")
            st.write(f"**Confidence**: {confidence:.2%}")
            st.write(f"**Frames Analyzed**: {frame_count}")
        
        # Clean up temporary file
        os.unlink(video_path)