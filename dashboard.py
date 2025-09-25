import streamlit as st
import torch
from timm import create_model
from torchvision import models, transforms
from PIL import Image
import pandas as pd

# -----------------------------
# CONFIG
# -----------------------------
device = torch.device("cpu")
st.set_page_config(page_title="Foot Ulcer Dashboard", layout="wide")

# Class names
ulcer_classes = ['no_ulcer', 'ulcer']
severity_classes = ['high', 'low', 'medium']

# Paths to models
ulcer_model_path = r"C:\Users\suthi\OneDrive\Desktop\foot_ulcer_project2\foot_ulcer_diagnosis_deit.pth"
severity_model_path = r"C:\Users\suthi\OneDrive\Desktop\foot_ulcer_project2\Severity_checking_model_complete.pth"

# -----------------------------
# HELPER FUNCTION TO LOAD MODEL
# -----------------------------
def load_model(model_path, architecture='deit_small_patch16_224', num_classes=2, is_timm=True):
    if is_timm:
        model = create_model(architecture, pretrained=False, num_classes=num_classes)
    else:
        model = models.resnet18(pretrained=False)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

    try:
        with torch.serialization.safe_globals([transforms.Compose]):
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    except:
        checkpoint = torch.load(model_path, map_location=device)

    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    return model

# -----------------------------
# LOAD MODELS
# -----------------------------
ulcer_model = load_model(ulcer_model_path, num_classes=len(ulcer_classes), is_timm=True)
severity_model = load_model(severity_model_path, num_classes=len(severity_classes), is_timm=True)

# -----------------------------
# IMAGE TRANSFORM
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# -----------------------------
# PREDICTION FUNCTIONS
# -----------------------------
def predict_ulcer(img: Image.Image):
    input_tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        output = ulcer_model(input_tensor)
        probs = torch.nn.functional.softmax(output, dim=1)[0]
        pred_idx = probs.argmax().item()
        pred_class = ulcer_classes[pred_idx]
        confidence = probs.max().item() * 100
    return pred_class, confidence, probs

def predict_severity(img: Image.Image):
    input_tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        output = severity_model(input_tensor)
        probs = torch.nn.functional.softmax(output, dim=1)[0]
        pred_idx = probs.argmax().item()
        pred_class = severity_classes[pred_idx]
        confidence = probs.max().item() * 100
    return pred_class, confidence, probs

# -----------------------------
# DASHBOARD UI
# -----------------------------
st.title("🦶 Foot Ulcer Diagnosis Dashboard")

# Sidebar for patient info
st.sidebar.header("Patient Information")
name = st.sidebar.text_input("Patient Name")
age = st.sidebar.number_input("Age", min_value=0, max_value=120, value=30)
diabetes = st.sidebar.selectbox("Diabetes", ["No", "Yes"])

# Upload foot image
uploaded_file = st.file_uploader("Upload Foot Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and diabetes == "Yes":
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=False, width=300)

    # -----------------------------
    # ULVER DETECTION
    # -----------------------------
    ulcer_result, ulcer_conf, ulcer_probs = predict_ulcer(img)

    # -----------------------------
    # SEVERITY PREDICTION
    # -----------------------------
    if ulcer_result == 'ulcer':
        severity_result, severity_conf, severity_probs = predict_severity(img)
    else:
        severity_result, severity_conf, severity_probs = None, None, None

    # -----------------------------
    # RESULTS DISPLAY
    # -----------------------------
    st.subheader("Patient Details")
    st.write(f"**Name:** {name}")
    st.write(f"**Age:** {age}")
    st.write(f"**Diabetes:** {diabetes}")

    st.subheader("Diagnosis Results")

    # Side-by-side layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Ulcer Detection**")
        st.write(f"Prediction: {ulcer_result} ({ulcer_conf:.2f}% confidence)")
        st.bar_chart({ulcer_classes[i]: float(ulcer_probs[i]) for i in range(len(ulcer_classes))})

    with col2:
        if severity_result:
            st.write("**Severity Prediction**")
            st.write(f"Severity: {severity_result} ({severity_conf:.2f}% confidence)")
            st.bar_chart({severity_classes[i]: float(severity_probs[i]) for i in range(len(severity_classes))})
        else:
            st.info("No ulcer detected → Severity check skipped.")
            
elif uploaded_file is not None and diabetes == "No":
    st.warning("Prediction skipped: Diabetes not present.")
