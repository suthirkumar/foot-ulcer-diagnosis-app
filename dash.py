import streamlit as st
import torch
from timm import create_model
from torchvision import models, transforms
from PIL import Image

# ----------------------------- CONFIG -----------------------------
device = torch.device("cpu")

# Class names
ulcer_classes = ['no_ulcer', 'ulcer']
severity_classes = ['high', 'low', 'medium']

# Paths to your models (update if needed)
ulcer_model_path = r"C:\Users\suthi\OneDrive\Desktop\foot_ulcer_project2\foot_ulcer_diagnosis_deit.pth"
severity_model_path = r"C:\Users\suthi\OneDrive\Desktop\foot_ulcer_project2\Severity_checking_model_complete.pth"

# ----------------------------- LOAD MODELS -----------------------------
@st.cache_resource
def load_model(model_path, architecture='resnet18', num_classes=2, is_timm=False):
    if is_timm:
        model = create_model(architecture, pretrained=False, num_classes=num_classes)
    else:
        model = models.resnet18(pretrained=False)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

    # ✅ Fix for PyTorch 2.6 default behavior
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    return model

ulcer_model = load_model(ulcer_model_path, architecture='deit_small_patch16_224',
                         num_classes=len(ulcer_classes), is_timm=True)
severity_model = load_model(severity_model_path, architecture='deit_small_patch16_224',
                            num_classes=len(severity_classes), is_timm=True)

# ----------------------------- IMAGE TRANSFORM -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ----------------------------- PREDICTION FUNCTIONS -----------------------------
def predict_ulcer(image):
    input_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = ulcer_model(input_tensor)
        probs = torch.nn.functional.softmax(output, dim=1)[0]
        pred_idx = probs.argmax().item()
        pred_class = ulcer_classes[pred_idx]
        confidence = probs.max().item() * 100
    return pred_class, confidence

def predict_severity(image):
    input_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = severity_model(input_tensor)
        probs = torch.nn.functional.softmax(output, dim=1)[0]
        pred_idx = probs.argmax().item()
        pred_class = severity_classes[pred_idx]
        confidence = probs.max().item() * 100
    return pred_class, confidence

# ----------------------------- UI STYLE -----------------------------
st.set_page_config(page_title="Diagnosis of Foot Ulcer", layout="centered")

st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #e6f0ff 0%, #f5faff 100%);
        font-family: 'Arial', sans-serif;
    }
    .card {
        background: white;
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 4px 25px rgba(0,0,0,0.08);
        text-align: center;
        margin: auto;
        width: 80%;
    }
    h1 {
        font-size: 2.2rem !important;
        font-weight: 700;
        color: #1c2e4a;
    }
    .subtitle {
        font-size: 1rem;
        color: #5a6a85;
        margin-bottom: 1rem;
    }
    .result-good {
        color: green;
        font-size: 1.2rem;
        font-weight: bold;
    }
    .result-bad {
        color: red;
        font-size: 1.2rem;
        font-weight: bold;
    }
    .summary-card {
        margin-top: 2rem;
        padding: 1.5rem;
        border-radius: 15px;
        font-size: 1.1rem;
        font-weight: 600;
    }
    .summary-good {
        background: #e6ffed;
        color: #087f23;
        border-left: 8px solid #087f23;
    }
    .summary-bad {
        background: #ffe6e6;
        color: #a70000;
        border-left: 8px solid #a70000;
    }
    </style>
""", unsafe_allow_html=True)

# ----------------------------- SESSION STATE -----------------------------
if "page" not in st.session_state:
    st.session_state.page = 1
if "patient_info" not in st.session_state:
    st.session_state.patient_info = {}

# ----------------------------- PAGE 1: PATIENT INFO -----------------------------
if st.session_state.page == 1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h1>Diagnosis of Foot Ulcer<br>using Deep Learning</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Fast & Accurate AI-powered analysis</p>", unsafe_allow_html=True)

    name = st.text_input("👤 Name")
    age = st.number_input("🎂 Age", min_value=1, max_value=120, step=1)
    diabetes = st.selectbox("💉 Diabetes Present?", ["Yes", "No"])

    if st.button("Next ➝"):
        if not name or not age:
            st.warning("⚠️ Please enter all details to proceed.")
        else:
            st.session_state.patient_info = {"name": name, "age": age, "diabetes": diabetes}
            st.session_state.page = 2
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

# ----------------------------- PAGE 2: IMAGE UPLOAD -----------------------------
elif st.session_state.page == 2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h1>Upload Foot Ulcer Image</h1>", unsafe_allow_html=True)
    st.markdown(f"<p class='subtitle'>Patient: <b>{st.session_state.patient_info['name']}</b> | "
                f"Age: <b>{st.session_state.patient_info['age']}</b> | "
                f"Diabetes: <b>{st.session_state.patient_info['diabetes']}</b></p>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("📤 Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Foot Image", use_column_width=True)

        if st.button("Analyze 🧠"):
            st.session_state.image = image
            st.session_state.page = 3
            st.rerun()

    if st.button("⬅ Back"):
        st.session_state.page = 1
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

# ----------------------------- PAGE 3: AI PREDICTION -----------------------------
elif st.session_state.page == 3:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h1>AI Diagnosis Result</h1>", unsafe_allow_html=True)

    image = st.session_state.image
    st.image(image, caption="Analyzed Image", use_column_width=True)

    ulcer_result, ulcer_conf = predict_ulcer(image)
    if ulcer_result == "ulcer":
        severity_result, severity_conf = predict_severity(image)
        st.markdown(f"<p class='result-bad'>🦶 Ulcer Detected ({ulcer_conf:.2f}% confidence)</p>", unsafe_allow_html=True)
        st.markdown(f"<p class='result-bad'>📊 Severity: {severity_result} ({severity_conf:.2f}% confidence)</p>", unsafe_allow_html=True)

        # Summary card (Bad)
        st.markdown("<div class='summary-card summary-bad'>🚨 Medical assistance is recommended immediately. Please consult a healthcare professional.</div>", unsafe_allow_html=True)

    else:
        st.markdown(f"<p class='result-good'>✅ No Ulcer Detected ({ulcer_conf:.2f}% confidence)</p>", unsafe_allow_html=True)

        # Summary card (Good)
        st.markdown("<div class='summary-card summary-good'>😊 No need to worry, you are safe. Keep maintaining good foot care habits.</div>", unsafe_allow_html=True)

    if st.button("⬅ Back"):
        st.session_state.page = 2
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)
