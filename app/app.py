import os
import sys
import tempfile

import cv2
import numpy as np
import streamlit as st
import torch


# ============================================================
# PATHS
# ============================================================

APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(APP_DIR, ".."))

sys.path.insert(0, APP_DIR)

from neural_codec import NeuralVideoCompressionModel


MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "checkpoints",
    "best_model.pth"
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

MODEL_WIDTH = 448
MODEL_HEIGHT = 256


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Neural Video Compression — Vox Lab",
    page_icon="◼",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# VOX MINIMALIST CUSTOM STYLING
# ============================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700;900&family=Inter:wght@400;500;700&display=swap');

    /* ======================================================
       GLOBAL STYLES & BACKGROUND
    ====================================================== */

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        background-color: #ffffff !important;
        color: #000000 !important;
        -webkit-font-smoothing: antialiased;
    }

    [data-testid="stAppViewContainer"],
    [data-testid="stMain"] {
        background: #ffffff !important;
    }

    .main .block-container {
        max-width: 1100px;
        padding-top: 40px;
        padding-bottom: 80px;
    }

    /* Hide standard Streamlit header elements */
    [data-testid="stHeader"] {
        background: #ffffff !important;
        border-bottom: 2px solid #000000;
    }

    [data-testid="stDecoration"],
    #MainMenu,
    footer {
        display: none !important;
        visibility: hidden !important;
    }

    /* ======================================================
       VOX TYPOGRAPHY & HERO
    ====================================================== */

    h1 {
        font-family: 'Space Grotesk', sans-serif !important;
        color: #000000 !important;
        font-size: 3.6rem !important;
        font-weight: 900 !important;
        letter-spacing: -0.04em !important;
        line-height: 1.05 !important;
        margin: 12px 0 16px 0 !important;
        text-transform: uppercase;
    }

    .eyebrow {
        display: inline-block;
        background: #000000;
        color: #fff000;
        font-family: 'Space Grotesk', sans-serif;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        padding: 4px 10px;
        border-radius: 0px;
    }

    .description {
        color: #222222;
        font-size: 1.1rem;
        line-height: 1.6;
        font-weight: 400;
        max-width: 760px;
        margin-bottom: 24px;
    }

    .vox-accent-bar {
        height: 6px;
        width: 100%;
        background: #000000;
        margin: 28px 0 40px 0;
        position: relative;
    }
    
    .vox-accent-bar::after {
        content: '';
        position: absolute;
        left: 0;
        top: 0;
        width: 120px;
        height: 100%;
        background: #fff000;
    }

    /* ======================================================
       SECTION HEADERS
    ====================================================== */

    .section-label {
        font-family: 'Space Grotesk', sans-serif;
        color: #000000;
        font-size: 0.85rem;
        font-weight: 700;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        margin: 32px 0 14px 0;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    .section-label::before {
        content: '//';
        color: #fff000;
        background: #000000;
        padding: 0 4px;
    }

    /* ======================================================
       UPLOADER (Sharp Minimal Grid)
    ====================================================== */

    [data-testid="stFileUploader"] {
        background: #ffffff !important;
        border: 2px solid #000000 !important;
        border-radius: 0px !important;
        padding: 16px !important;
    }

    [data-testid="stFileUploaderDropzone"] {
        background: #fafafa !important;
        border: 1px dashed #000000 !important;
        border-radius: 0px !important;
    }

    [data-testid="stFileUploaderDropzoneInstructions"] span,
    [data-testid="stFileUploaderDropzoneInstructions"] small {
        color: #000000 !important;
        font-family: 'Space Grotesk', sans-serif !important;
    }

    [data-testid="stBaseButton-secondary"] {
        color: #000000 !important;
        background: #fff000 !important;
        border: 2px solid #000000 !important;
        border-radius: 0px !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
    }

    [data-testid="stBaseButton-secondary"]:hover {
        background: #000000 !important;
        color: #ffffff !important;
    }

    /* ======================================================
       METRICS & CONTAINERS
    ====================================================== */

    [data-testid="stMetric"] {
        background: #ffffff !important;
        border: 2px solid #000000 !important;
        border-radius: 0px !important;
        padding: 16px 20px !important;
        box-shadow: 4px 4px 0px #000000;
    }

    [data-testid="stMetricLabel"] {
        color: #000000 !important;
        font-family: 'Space Grotesk', sans-serif !important;
        font-size: 0.72rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.12em !important;
        text-transform: uppercase !important;
    }

    [data-testid="stMetricValue"] {
        color: #000000 !important;
        font-family: 'Space Grotesk', sans-serif !important;
        font-size: 1.75rem !important;
        font-weight: 900 !important;
    }

    /* ======================================================
       BUTTONS (Vox Punchy Action Style)
    ====================================================== */

    .stButton > button {
        width: 100%;
        min-height: 52px;
        border-radius: 0px !important;
        border: 2px solid #000000 !important;
        background: #000000 !important;
        color: #ffffff !important;
        font-family: 'Space Grotesk', sans-serif !important;
        font-size: 1rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.1em !important;
        text-transform: uppercase !important;
        transition: all 0.15s ease-in-out;
        box-shadow: 4px 4px 0px #fff000;
    }

    .stButton > button:hover {
        background: #fff000 !important;
        color: #000000 !important;
        border-color: #000000 !important;
        box-shadow: 4px 4px 0px #000000;
    }

    .stDownloadButton > button {
        width: 100%;
        min-height: 52px;
        border-radius: 0px !important;
        border: 2px solid #000000 !important;
        background: #ffffff !important;
        color: #000000 !important;
        font-family: 'Space Grotesk', sans-serif !important;
        font-size: 1rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.08em !important;
        text-transform: uppercase !important;
        box-shadow: 4px 4px 0px #000000;
    }

    .stDownloadButton > button:hover {
        background: #000000 !important;
        color: #fff000 !important;
    }

    /* ======================================================
       PROGRESS & STATUS
    ====================================================== */

    .stProgress > div > div > div > div {
        background-color: #fff000 !important;
        border-top: 2px solid #000000;
        border-bottom: 2px solid #000000;
    }

    [data-testid="stAlert"] {
        border-radius: 0px !important;
        border: 2px solid #000000 !important;
        background: #ffffff !important;
        color: #000000 !important;
    }

    /* ======================================================
       DIVIDER & FOOTER
    ====================================================== */

    hr {
        border: none !important;
        border-top: 2px solid #000000 !important;
        margin: 40px 0 !important;
    }

    .fine-print {
        color: #000000;
        font-family: 'Space Grotesk', sans-serif;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        border-left: 3px solid #fff000;
        padding-left: 10px;
    }

    .video-container-label {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        margin-bottom: 8px;
        background: #000000;
        color: #ffffff;
        display: inline-block;
        padding: 2px 8px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# MODEL LOAD
# ============================================================

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model checkpoint not found:\n{MODEL_PATH}"
        )

    model = NeuralVideoCompressionModel(
        latent_channels=64
    ).to(DEVICE)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    return model


# ============================================================
# PREPROCESS
# ============================================================

def preprocess_frame(frame):
    frame_rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    frame_rgb = cv2.resize(
        frame_rgb,
        (MODEL_WIDTH, MODEL_HEIGHT),
        interpolation=cv2.INTER_AREA
    )

    tensor = (
        torch.from_numpy(frame_rgb)
        .float()
        / 255.0
    )

    tensor = tensor.permute(
        2, 0, 1
    ).unsqueeze(0)

    return tensor.to(DEVICE)


# ============================================================
# RESTORE
# ============================================================

def restore_frame(tensor, width, height):
    image = (
        tensor[0]
        .detach()
        .cpu()
        .permute(1, 2, 0)
        .numpy()
    )

    image = np.clip(
        image * 255.0,
        0,
        255
    ).astype(np.uint8)

    image = cv2.resize(
        image,
        (width, height),
        interpolation=cv2.INTER_CUBIC
    )

    return cv2.cvtColor(
        image,
        cv2.COLOR_RGB2BGR
    )


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="eyebrow">DEEP LEARNING // COMPUTER VISION</div>',
    unsafe_allow_html=True
)

st.title("Neural Video Compression")

st.markdown(
    """
    <div class="description">
        A learned frame-reconstruction system utilizing neural motion estimation, 
        residual coding, and high-density latent representation.
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    '<div class="vox-accent-bar"></div>',
    unsafe_allow_html=True
)


# ============================================================
# INPUT
# ============================================================

st.markdown(
    '<div class="section-label">01 / INPUT VIDEO</div>',
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Choose video",
    type=["mp4", "webm", "avi", "mov"],
    label_visibility="collapsed"
)

if uploaded_file is None:
    st.markdown(
        """
        <div class="fine-print">
            <br>
            Upload a video file to run frame reconstruction through the trained neural model.
        </div>
        """,
        unsafe_allow_html=True
    )
    st.stop()


# ============================================================
# SAVE INPUT
# ============================================================

suffix = os.path.splitext(uploaded_file.name)[1]

input_file = tempfile.NamedTemporaryFile(
    delete=False,
    suffix=suffix
)

input_file.write(uploaded_file.getvalue())
input_file.close()

input_path = input_file.name


# ============================================================
# READ VIDEO INFORMATION
# ============================================================

cap = cv2.VideoCapture(input_path)

if not cap.isOpened():
    st.error("Could not open the uploaded video.")
    st.stop()

fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 0:
    fps = 30.0

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

cap.release()


# ============================================================
# ORIGINAL VIDEO
# ============================================================

st.markdown(
    '<div class="video-container-label">SOURCE MEDIA</div>',
    unsafe_allow_html=True
)

st.video(uploaded_file.getvalue())


# ============================================================
# INFORMATION METRICS
# ============================================================

st.markdown(
    '<div class="section-label">02 / VIDEO SPECIFICATIONS</div>',
    unsafe_allow_html=True
)

c1, c2, c3 = st.columns(3)

with c1:
    st.metric("Resolution", f"{width} × {height}")

with c2:
    st.metric("Frame Rate", f"{fps:.2f} FPS")

with c3:
    st.metric("Total Frames", f"{frame_count:,}")

st.write("")


# ============================================================
# MODEL CONFIGURATION METRICS
# ============================================================

st.markdown(
    '<div class="section-label">03 / ARCHITECTURE PARAMETERS</div>',
    unsafe_allow_html=True
)

m1, m2, m3 = st.columns(3)

with m1:
    st.metric("Architecture", "CNN + Autoencoder")

with m2:
    st.metric("Model Grid", "448 × 256")

with m3:
    st.metric("Weights Status", "Trained Scratch")

st.write("")


# ============================================================
# RUN COMPRESSION
# ============================================================

run = st.button("EXECUTE NEURAL COMPRESSION")

if not run:
    st.stop()


# ============================================================
# LOAD MODEL
# ============================================================

with st.spinner("Loading neural network weights..."):
    try:
        model = load_model()
    except Exception as error:
        st.error(f"Model loading failed: {error}")
        st.stop()


# ============================================================
# OUTPUT FILE PREPARATION
# ============================================================

output_file = tempfile.NamedTemporaryFile(
    delete=False,
    suffix=".mp4"
)
output_file.close()
output_path = output_file.name

cap = cv2.VideoCapture(input_path)
if not cap.isOpened():
    st.error("Could not reopen the uploaded video.")
    st.stop()

writer = cv2.VideoWriter(
    output_path,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)


# ============================================================
# FIRST FRAME PROCESS
# ============================================================

success, frame = cap.read()
if not success:
    cap.release()
    writer.release()
    st.error("The uploaded video contains no readable frames.")
    st.stop()

writer.write(frame)
previous_tensor = preprocess_frame(frame)
processed_frames = 1


# ============================================================
# PROCESSING LOOP
# ============================================================

st.markdown(
    '<div class="section-label">04 / FRAME RECONSTRUCTION PROGRESS</div>',
    unsafe_allow_html=True
)

progress = st.progress(0)
progress_text = st.empty()

with torch.no_grad():
    while True:
        success, frame = cap.read()
        if not success:
            break

        current_tensor = preprocess_frame(frame)
        outputs = model(previous_tensor, current_tensor)

        reconstructed = restore_frame(
            outputs["reconstructed_frame"],
            width,
            height
        )

        writer.write(reconstructed)
        previous_tensor = current_tensor
        processed_frames += 1

        progress_value = min(
            processed_frames / max(frame_count, 1),
            1.0
        )

        progress.progress(progress_value)
        progress_text.caption(
            f"STATUS: {processed_frames:,} / {frame_count:,} FRAMES PROCESSED"
        )

cap.release()
writer.release()


# ============================================================
# RESULTS & OUTPUT
# ============================================================

st.success(
    f"COMPRESSION COMPLETE · {processed_frames:,} FRAMES RECONSTRUCTED"
)

st.markdown(
    '<div class="section-label">05 / RECONSTRUCTED OUTPUT</div>',
    unsafe_allow_html=True
)

st.video(output_path)


# ============================================================
# OUTPUT METRICS
# ============================================================

c1, c2, c3 = st.columns(3)

with c1:
    st.metric("Output Resolution", f"{width} × {height}")

with c2:
    st.metric("Processed Frames", f"{processed_frames:,}")

with c3:
    st.metric("Target Rate", f"{fps:.2f} FPS")

st.write("")


# ============================================================
# DOWNLOAD BUTTON
# ============================================================

with open(output_path, "rb") as file:
    st.download_button(
        "DOWNLOAD RECONSTRUCTED VIDEO",
        data=file,
        file_name="reconstructed_video.mp4",
        mime="video/mp4"
    )

st.divider()

st.markdown(
    """
    <div class="fine-print">
        NEURAL VIDEO RECONSTRUCTION CODEC // INFERENCE MODE ONLY // TRAINED WITHOUT PRETRAINED WEIGHTS
    </div>
    """,
    unsafe_allow_html=True
)