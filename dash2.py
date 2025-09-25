import streamlit as st
import torch
from timm import create_model
from torchvision import transforms, models
from PIL import Image

# -----------------------------
# CONFIGuration 
# -----------------------------
st.set_page_config(page_title="Foot Ulcer Diagnosis", layout="wide")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ulcer_classes = ['no_ulcer', 'ulcer']
severity_classes = ['high', 'medium', 'low']

ulcer_model_path = r"C:\Users\suthi\OneDrive\Desktop\foot_ulcer_project2\foot_ulcer_diagnosis_deit.pth"
severity_model_path = r"C:\Users\suthi\OneDrive\Desktop\foot_ulcer_project2\Severity_checking_model_complete.pth"

# -----------------------------
# HELPER FUNCTIONS
# -----------------------------
def load_model(model_path, architecture='deit_small_patch16_224', num_classes=2, is_timm=True):
    """Load model weights safely using weights_only=False (trusted source)."""
    if is_timm:
        model = create_model(architecture, pretrained=False, num_classes=num_classes)
    else:
        model = models.resnet18(pretrained=False)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

    # Use weights_only=False to fully load checkpoint (trusted source)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # If checkpoint contains state dict
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()
    return model

# -----------------------------
# LOAD MODELS ONCE
# -----------------------------
@st.cache_resource
def get_models():
    ulcer_model = load_model(ulcer_model_path, num_classes=len(ulcer_classes))
    severity_model = load_model(severity_model_path, num_classes=len(severity_classes))
    return ulcer_model, severity_model

ulcer_model, severity_model = get_models()

# -----------------------------
# IMAGE TRANSFORMATION
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
def predict_ulcer(img):
    input_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = ulcer_model(input_tensor)
        probs = torch.nn.functional.softmax(output, dim=1)[0]
        idx = probs.argmax().item()
        return ulcer_classes[idx], probs[idx].item() * 100, probs

def predict_severity(img):
    input_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = severity_model(input_tensor)
        probs = torch.nn.functional.softmax(output, dim=1)[0]
        idx = probs.argmax().item()
        return severity_classes[idx], probs[idx].item() * 100, probs

# -----------------------------
# SESSION STATE INIT
# -----------------------------
if 'page' not in st.session_state:
    st.session_state.page = 1
if 'patient_info' not in st.session_state:
    st.session_state.patient_info = {}
if 'uploaded_img' not in st.session_state:
    st.session_state.uploaded_img = None
if 'diagnosis' not in st.session_state:
    st.session_state.diagnosis = {}

# -----------------------------
# PAGE NAVIGATION
# -----------------------------
def next_page():
    st.session_state.page += 1

def prev_page():
    st.session_state.page -= 1

# -----------------------------
# PAGE 1: PATIENT DETAILS
# -----------------------------
if st.session_state.page == 1:
    st.title("🦶 Foot Ulcer Diagnosis - Patient Details")
    name = st.text_input("Patient Name")
    age = st.number_input("Age", min_value=0, max_value=120, value=30)
    diabetes = st.selectbox("Diabetes Present?", ["No", "Yes"])

    if st.button("Next"):
        if name and diabetes:
            st.session_state.patient_info = {'name': name, 'age': age, 'diabetes': diabetes}
            next_page()
        else:
            st.warning("Please fill in all details!")

# -----------------------------
# PAGE 2: UPLOAD FOOT IMAGE
# -----------------------------
elif st.session_state.page == 2:
    st.title("🦶 Foot Ulcer Diagnosis - Upload Foot Image")
    st.write("Upload foot image for diagnosis:")

    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Uploaded Image", use_column_width=False, width=300)
        st.session_state.uploaded_img = img

    col1, col2 = st.columns(2)
    with col1:
        if st.button("⬅ Back"):
            prev_page()
    with col2:
        if uploaded_file and st.session_state.patient_info.get('diabetes') == "Yes":
            st.button("Next", on_click=next_page)
        elif uploaded_file and st.session_state.patient_info.get('diabetes') == "No":
            st.warning("Prediction skipped: Diabetes not present.")

# -----------------------------
# PAGE 3: RESULTS
# -----------------------------
elif st.session_state.page == 3:
    st.title("🦶 Foot Ulcer Diagnosis - Results")
    img = st.session_state.uploaded_img
    st.image(img, caption="Uploaded Image", width=300)

    st.subheader("Patient Details")
    for k, v in st.session_state.patient_info.items():
        st.write(f"**{k.capitalize()}:** {v}")

    # ULVER DETECTION
    ulcer_result, ulcer_conf, ulcer_probs = predict_ulcer(img)

    # SEVERITY PREDICTION
    if ulcer_result == 'ulcer':
        severity_result, severity_conf, severity_probs = predict_severity(img)
    else:
        severity_result, severity_conf, severity_probs = None, None, None

    # RESULTS DISPLAY
    st.subheader("Diagnosis Results")
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

    # SUMMARY CARD
    st.subheader("Summary")
    if ulcer_result == 'ulcer':
        st.markdown(
            "<div style='background-color:#FF6961; padding:15px; border-radius:10px; color:white; text-align:center'>Medical assistance recommended immediately.</div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            "<div style='background-color:#77DD77; padding:15px; border-radius:10px; color:white; text-align:center'>No need to worry, maintain foot care.</div>",
            unsafe_allow_html=True
        )

    # BACK / RESET
    col1, col2 = st.columns(2)
    with col1:
        st.button("⬅ Back", on_click=prev_page)
    
    with col2:
        if st.session_state.get('rerun', False):
            # Reset the flag first
            st.session_state['rerun'] = False
            # Then rerun the script
            st.experimental_rerun()
        
        # Button to trigger rerun
        st.button("Reset", on_click=lambda: st.session_state.update({'rerun': True}))



            