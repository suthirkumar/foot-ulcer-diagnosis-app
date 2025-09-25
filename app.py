import streamlit as st
import torch
from timm import create_model
from torchvision import models, transforms
from PIL import Image

# -----------------------------
# CONFIG
# -----------------------------
device = torch.device("cpu")  # CPU is fine for inference
st.set_page_config(page_title="Foot Ulcer Diagnosis", layout="centered")

# Class names
ulcer_classes = ['no_ulcer', 'ulcer']
severity_classes = ['high', 'low', 'medium']

# Paths to your models
ulcer_model_path = r"C:\Users\suthi\OneDrive\Desktop\foot_ulcer_project2\foot_ulcer_diagnosis_deit.pth"
severity_model_path = r"C:\Users\suthi\OneDrive\Desktop\foot_ulcer_project2\Severity_checking_model_complete.pth"

# -----------------------------
# HELPER FUNCTION TO LOAD MODEL
# -----------------------------
def load_model(model_path, architecture='resnet18', num_classes=2, is_timm=False):
    if is_timm:
        model = create_model(architecture, pretrained=False, num_classes=num_classes)
    else:
        model = models.resnet18(pretrained=False)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

    # Load checkpoint safely
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
ulcer_model = load_model(ulcer_model_path, architecture='deit_small_patch16_224', num_classes=len(ulcer_classes), is_timm=True)
severity_model = load_model(severity_model_path, architecture='deit_small_patch16_224', num_classes=len(severity_classes), is_timm=True)

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
# STREAMLIT UI
# -----------------------------
st.title("🦶 Foot Ulcer Diagnosis & Severity Detection")
st.write("Upload a foot image to check for ulcers and severity (if present).")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Step 1: Ulcer Detection
    ulcer_result, ulcer_conf, ulcer_probs = predict_ulcer(img)
    st.subheader("Ulcer Detection")
    st.write(f"**Prediction:** {ulcer_result} ({ulcer_conf:.2f}% confidence)")

    # Step 2: Severity Check (if ulcer present)
    if ulcer_result == 'ulcer':
        severity_result, severity_conf, severity_probs = predict_severity(img)
        st.subheader("Severity Prediction")
        st.write(f"**Severity:** {severity_result} ({severity_conf:.2f}% confidence)")

        st.bar_chart({severity_classes[i]: float(severity_probs[i]) for i in range(len(severity_classes))})
    else:
        st.info("No ulcer detected → Severity check skipped.")
