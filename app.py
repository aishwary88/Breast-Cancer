import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

from pathlib import Path
from PIL import Image
import numpy as np
import cv2

from gradcam import GradCAM, generate_gradcam_figure

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Breast Cancer Classification",
    page_icon="🩺",
    layout="wide",
)

# ============================================================
# CONFIG
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

IMAGE_SIZE = 448

# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    model = models.convnext_tiny(weights=None)

    # Grayscale input modification
    model.features[0][0] = nn.Conv2d(
        in_channels=1,
        out_channels=96,
        kernel_size=4,
        stride=4,
    )

    # Binary classifier head
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_features, 1),
    )

    model = model.to(DEVICE)

    checkpoint = torch.load(
        "best_fold_0.pth",
        map_location=DEVICE,
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model


model = load_model()

# ============================================================
# CLAHE
# ============================================================

def apply_clahe(img):
    img   = np.array(img)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    img   = clahe.apply(img)
    return Image.fromarray(img)

# ============================================================
# TRANSFORMS
# ============================================================

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])

# ============================================================
# UI — HEADER
# ============================================================

st.title("Breast Cancer Classification")
st.markdown(
    "Upload a mammogram image or click a sample below. "
    "The model predicts **Benign** or **Malignant** and shows "
    "a **Grad-CAM heatmap** of the regions it focused on."
)
st.markdown("---")

# ============================================================
# SIDEBAR — sample buttons + model info
# ============================================================

with st.sidebar:
    st.header("Try a Sample Image")

    sample_files = {
        "Malignant Sample": "sample_malignant.jpg",
        "Benign Sample":    "sample_benign.jpg",
    }

    # Show button only — no thumbnail
    for label, path in sample_files.items():
        if Path(path).exists():
            if st.button(f"Use {label}", key=label):
                st.session_state["sample_path"] = path
        else:
            st.warning(f"{path} not found")

    st.markdown("---")
    st.header("Model Info")
    st.write(f"**Architecture:** ConvNeXt-Tiny")
    st.write(f"**Input size:** {IMAGE_SIZE}x{IMAGE_SIZE}")
    st.write(f"**Device:** {DEVICE}")
    st.write(f"**Preprocessing:** CLAHE + Normalize")

# ============================================================
# DETERMINE ACTIVE IMAGE SOURCE
# ============================================================

# Priority: uploaded file > clicked sample > nothing
uploaded_file = st.file_uploader(
    "Or upload your own mammogram image",
    type=["png", "jpg", "jpeg"],
)

active_image_pil  = None
active_image_name = None

if uploaded_file is not None:
    # Uploaded file takes priority — also clear any sample selection
    st.session_state.pop("sample_path", None)
    active_image_pil  = Image.open(uploaded_file).convert("L")
    active_image_name = uploaded_file.name

elif "sample_path" in st.session_state:
    path = st.session_state["sample_path"]
    if Path(path).exists():
        active_image_pil  = Image.open(path).convert("L")
        active_image_name = path

# ============================================================
# PROCESS & DISPLAY
# ============================================================

if active_image_pil is not None:

    orig_image = active_image_pil.copy()

    st.markdown(f"**Image:** `{active_image_name}`")
    st.markdown("---")

    # Preprocess
    image_clahe  = apply_clahe(active_image_pil)
    input_tensor = transform(image_clahe).unsqueeze(0).to(DEVICE)

    # ========================================================
    # PREDICTION
    # ========================================================

    with torch.no_grad():
        output      = model(input_tensor)
        probability = torch.sigmoid(output).item()

    prediction = "MALIGNANT" if probability > 0.5 else "BENIGN"
    color      = "red" if prediction == "MALIGNANT" else "green"

    # ========================================================
    # LAYOUT — two columns
    # ========================================================

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Uploaded Image")
        st.image(active_image_pil, use_container_width=True, clamp=True)

    with col2:
        st.subheader("Prediction Result")

        # Big prediction label
        st.markdown(
            f"<h2 style='color:{color};'>{prediction}</h2>",
            unsafe_allow_html=True,
        )

        # Probability bar
        st.metric(
            label="Malignant Probability",
            value=f"{probability:.4f}",
            delta=f"{'High risk' if probability > 0.5 else 'Low risk'}",
            delta_color="inverse",
        )

        st.progress(float(probability))

        st.markdown("---")
        st.caption(
            "⚠️ This tool is for research purposes only. "
            "Always consult a qualified radiologist for diagnosis."
        )

    # ========================================================
    # GRAD-CAM
    # ========================================================

    st.markdown("---")
    st.subheader("Grad-CAM Visualization")
    st.write(
        "The heatmap shows which regions of the mammogram "
        "influenced the prediction. **Red = high attention**, "
        "Blue = low attention."
    )

    with st.spinner("Generating Grad-CAM heatmap..."):
        try:
            # GradCAM needs gradients — use a fresh tensor
            grad_tensor = transform(image_clahe).unsqueeze(0).to(DEVICE)
            gradcam     = GradCAM(model)
            cam, prob   = gradcam.generate(grad_tensor)

            fig = generate_gradcam_figure(orig_image, cam, prob)
            st.pyplot(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Grad-CAM failed: {e}")
