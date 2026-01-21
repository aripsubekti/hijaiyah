import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import json
from pathlib import Path
from collections import OrderedDict

st.set_page_config(page_title="Deteksi Isyarat Hijaiyah", layout="centered")

@st.cache_data
def load_class_names():
    base = Path(__file__).resolve().parent
    with open(base / "class_names.json", "r", encoding="utf-8") as f:
        return json.load(f)

class_names = load_class_names()

@st.cache_resource
def load_model():
    base = Path(__file__).resolve().parent
    model_path = base / "efficientnetb1_best_model_2s.pth"

    model = models.efficientnet_b1(weights=None)

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 29)

    state = torch.load(model_path, map_location="cpu")

    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]

    if isinstance(state, dict) and any(k.startswith("module.") for k in state.keys()):
        state = OrderedDict((k.replace("module.", ""), v) for k, v in state.items())

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        st.warning("State dict mismatch terdeteksi (cek arsitektur / jumlah kelas).")
        st.write("Missing keys:", missing[:30], "..." if len(missing) > 30 else "")
        st.write("Unexpected keys:", unexpected[:30], "..." if len(unexpected) > 30 else "")

    model.eval()
    return model

model = load_model()

st.title("Deteksi Bahasa Isyarat Hijaiyah Indonesia (EfficientNet-B1)")
st.write("Silakan upload gambar tangan dengan isyarat huruf hijaiyah.")

col1, col2 = st.columns(2)
uploaded_file = col1.file_uploader("Upload", type=["jpg", "png", "jpeg"])

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    col1.image(image, caption="Gambar Isyarat", use_container_width=True)

    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(input_tensor)
        predicted_idx = int(outputs.argmax(dim=1).item())

    info = class_names[str(predicted_idx)]
    huruf = info["huruf"]
    latin = info["latin"]

    with col2:
        st.markdown("### Prediction")
        st.markdown(f"<div style='font-size:80px; text-align:center'>{huruf}</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='text-align:center; font-size:20px'>{latin}</div>", unsafe_allow_html=True)
