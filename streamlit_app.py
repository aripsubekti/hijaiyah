import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision import models
from PIL import Image
import json

# --- Load Model ---
class CNNModel(nn.Module):
    def __init__(self, num_classes):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(32 * 53 * 53, num_classes)  # Sesuaikan input size FC dengan output dari conv+pool

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

@st.cache_resource
def load_model():
    num_classes = 30
    model = models.resnet50(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load("efficientnetb1_best_model_2s.pth", map_location="cpu"))
    model.eval()
    return model

@st.cache_data
def load_class_names():
    with open('class_names.json', 'r') as f:
        return json.load(f)

# --- Load Label ---
class_names = load_class_names()
model = load_model()

# --- UI Layout ---
st.set_page_config(page_title="Deteksi Isyarat Hijaiyah", layout="centered")

st.title("Deteksi Bahasa Isyarat Hijaiyah Indonesia dengan CNN")
st.write("Silakan upload gambar tangan dengan isyarat huruf hijaiyah.")

col1, col2 = st.columns(2)

uploaded_file = col1.file_uploader("Upload", type=['jpg', 'png', 'jpeg'])

# --- Preprocessing ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    col1.image(image, caption='Gambar Isyarat', use_container_width=True)

    input_tensor = transform(image).unsqueeze(0)

    # --- Prediction ---
    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)
        class_idx = predicted.item()
        huruf = class_names[str(class_idx)]['huruf']
        latin = class_names[str(class_idx)]['latin']

    # --- Display Prediction ---
    with col2:
        st.markdown("### Prediction")
        st.markdown(f"<div style='font-size:80px; text-align:center'>{huruf}</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='text-align:center; font-size:20px'>{latin}</div>", unsafe_allow_html=True)
