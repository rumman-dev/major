import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import io
import os
import streamlit as st
import time

# --- Configuration ---
NUM_CLASSES = 3
CLASS_NAMES = ['Cyclone', 'Flood', 'Wildfire']
MODEL_PATH = 'best_student_model.pth'

# --- StudentCNN Model ---
class StudentCNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(StudentCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flattened_size = 32 * 28 * 28
        self.fc = nn.Linear(self.flattened_size, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# --- Load Model ---
device = torch.device("cpu")
model = StudentCNN(num_classes=NUM_CLASSES)

try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    st.stop()

# --- Image Transformations ---
inference_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# --- Streamlit App ---
st.set_page_config(page_title="Disaster Classifier", layout="centered")
st.title("🌪️ Disaster Classification using CNN")
st.markdown("Upload an image of a disaster to classify it as **Cyclone**, **Flood**, or **Wildfire**.")

uploaded_file = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Uploaded Image', use_column_width=True)

        input_tensor = inference_transform(image).unsqueeze(0).to(device)

        # Start timer
        start_time = time.time()

        with torch.no_grad():
            output = model(input_tensor)
            probabilities = F.softmax(output, dim=1)[0]
            predicted_idx = torch.argmax(probabilities).item()
            predicted_class = CLASS_NAMES[predicted_idx]
            confidence = probabilities[predicted_idx].item() * 100

        # End timer
        end_time = time.time()
        response_time = end_time - start_time

        st.success(f"✅ Prediction: **{predicted_class}** with confidence **{confidence:.2f}%**")
        st.info(f"🕒 Response Time: **{response_time:.4f} seconds**")

    except Exception as e:
        st.error(f"Error processing image: {e}")
