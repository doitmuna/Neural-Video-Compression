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
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Neural Video Compression",
    page_icon="◼",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# CUSTOM DESIGN
# ============================================================

st.markdown(
    """
    <style>

    /* =========================================================
       GLOBAL
    ========================================================= */

    [data-testid="stAppViewContainer"],
    [data-testid="stMain"],
    .main {
        background: #ffffff !important;
    }

    .main .block-container {
        max-width: 1180px;
        padding-top: 42px;
        padding-bottom: 80px;
    }

    html,
    body,
    [class*="css"] {
        font-family:
            -apple-system,
            BlinkMacSystemFont,
            "Segoe UI",
            Helvetica,
            Arial,
            sans-serif;
    }


    /* =========================================================
       STREAMLIT CHROME
    ========================================================= */

    [data-testid="stHeader"] {
        background: #ffffff !important;
        border-bottom: 1px solid #e5e5e5;
    }

    [data-testid="stDecoration"] {
        display: none;
    }

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }


    /* =========================================================
       TYPOGRAPHY
    ========================================================= */

    h1 {
        color: #111111 !important;

        font-family:
            Georgia,
            "Times New Roman",
            serif !important;

        font-size: 4.1rem !important;
        font-weight: 700 !important;

        letter-spacing: -0.055em !important;
        line-height: 0.98 !important;

        margin: 0 !important;
    }

    h2 {
        color: #111111 !important;

        font-family:
            Georgia,
            "Times New Roman",
            serif !important;

        font-size: 1.8rem !important;
        font-weight: 700 !important;

        letter-spacing: -0.025em !important;
    }

    h3 {
        color: #111111 !important;
    }

    p {
        color: #444444;
    }


    /* =========================================================
       TOP BRAND BAR
    ========================================================= */

    .brand-bar {
        display: flex;
        align-items: center;
        justify-content: space-between;

        padding-bottom: 15px;
        margin-bottom: 48px;

        border-bottom: 2px solid #111111;
    }

    .brand {
        color: #111111;

        font-size: 0.78rem;
        font-weight: 800;

        letter-spacing: 0.16em;
        text-transform: uppercase;
    }

    .brand-dot {
        color: #d71920;
    }

    .brand-meta {
        color: #777777;

        font-size: 0.68rem;
        font-weight: 600;

        letter-spacing: 0.08em;
        text-transform: uppercase;
    }


    /* =========================================================
       HERO
    ========================================================= */

    .eyebrow {
        color: #d71920;

        font-size: 0.72rem;
        font-weight: 800;

        letter-spacing: 0.13em;
        text-transform: uppercase;

        margin-bottom: 13px;
    }

    .description {
        color: #555555;

        font-size: 1.02rem;
        line-height: 1.65;

        margin-top: 20px;

        max-width: 700px;
    }

    .hero-rule {
        height: 5px;
        width: 76px;

        background: #d71920;

        margin: 28px 0 55px 0;
    }


    /* =========================================================
       SECTION HEADERS
    ========================================================= */

    .section-header {
        display: flex;
        align-items: baseline;

        gap: 12px;

        border-top: 1px solid #111111;

        padding-top: 13px;

        margin-top: 46px;
        margin-bottom: 20px;
    }

    .section-number {
        color: #d71920;

        font-size: 0.68rem;
        font-weight: 800;

        letter-spacing: 0.08em;
    }

    .section-title {
        color: #111111;

        font-size: 0.72rem;
        font-weight: 800;

        letter-spacing: 0.12em;
        text-transform: uppercase;
    }


    /* =========================================================
       FILE UPLOADER
    ========================================================= */

    [data-testid="stFileUploader"] {
        background: #fafafa !important;

        border: 1px solid #d9d9d9 !important;

        border-radius: 0 !important;

        padding: 5px !important;
    }

    [data-testid="stFileUploaderDropzone"] {
        background: #ffffff !important;

        border: 1px dashed #bdbdbd !important;

        border-radius: 0 !important;

        min-height: 145px;
    }

    [data-testid="stFileUploaderDropzone"]:hover {
        border-color: #111111 !important;
    }

    [data-testid="stFileUploaderDropzoneInstructions"] {
        color: #444444 !important;
    }

    [data-testid="stFileUploaderDropzoneInstructions"] span {
        color: #444444 !important;
    }

    [data-testid="stFileUploaderDropzoneInstructions"] small {
        color: #888888 !important;
    }

    [data-testid="stBaseButton-secondary"] {
        color: #111111 !important;

        background: #ffffff !important;

        border: 1px solid #111111 !important;

        border-radius: 0 !important;
    }


    /* =========================================================
       VIDEO
    ========================================================= */

    .video-label {
        color: #777777;

        font-size: 0.67rem;
        font-weight: 800;

        letter-spacing: 0.11em;
        text-transform: uppercase;

        margin: 8px 0 10px 0;
    }

    video {
        border-radius: 0 !important;
    }


    /* =========================================================
       METRICS
    ========================================================= */

    [data-testid="stMetric"] {
        background: #ffffff !important;

        border-top: 1px solid #111111 !important;
        border-bottom: 1px solid #d9d9d9 !important;

        border-left: none !important;
        border-right: none !important;

        border-radius: 0 !important;

        padding: 16px 0 14px 0 !important;
    }

    [data-testid="stMetricLabel"] {
        color: #777777 !important;

        font-size: 0.65rem !important;
        font-weight: 800 !important;

        letter-spacing: 0.10em;
        text-transform: uppercase;
    }

    [data-testid="stMetricValue"] {
        color: #111111 !important;

        font-family:
            Georgia,
            "Times New Roman",
            serif !important;

        font-size: 1.55rem !important;
        font-weight: 700 !important;

        letter-spacing: -0.025em;
    }


    /* =========================================================
       PRIMARY BUTTON
    ========================================================= */

    .stButton > button {
        width: 100%;

        min-height: 48px;

        border-radius: 0 !important;

        border: 1px solid #111111 !important;

        background: #111111 !important;

        color: #ffffff !important;

        font-size: 0.73rem !important;
        font-weight: 800 !important;

        letter-spacing: 0.08em;

        text-transform: uppercase;

        transition:
            background 0.15s ease,
            border-color 0.15s ease;
    }

    .stButton > button:hover {
        background: #d71920 !important;
        border-color: #d71920 !important;
    }


    /* =========================================================
       DOWNLOAD BUTTON
    ========================================================= */

    .stDownloadButton > button {
        width: 100%;

        min-height: 48px;

        border-radius: 0 !important;

        border: 1px solid #111111 !important;

        background: #ffffff !important;

        color: #111111 !important;

        font-size: 0.73rem !important;
        font-weight: 800 !important;

        letter-spacing: 0.08em;

        text-transform: uppercase;
    }

    .stDownloadButton > button:hover {
        background: #111111 !important;
        color: #ffffff !important;
    }


    /* =========================================================
       PROGRESS
    ========================================================= */

    [data-testid="stProgress"] {
        margin-top: 10px;
        margin-bottom: 8px;
    }

    [data-testid="stProgressBar"] {
        background: #eeeeee !important;
        border-radius: 0 !important;
    }


    /* =========================================================
       ALERTS
    ========================================================= */

    [data-testid="stAlert"] {
        border-radius: 0 !important;
        border-left-width: 4px !important;
    }


    /* =========================================================
       DIVIDER
    ========================================================= */

    hr {
        border: none !important;

        border-top: 1px solid #dddddd !important;

        margin: 42px 0 !important;
    }


    /* =========================================================
       SMALL TEXT
    ========================================================= */

    .fine-print {
        color: #888888;

        font-size: 0.70rem;

        line-height: 1.55;

        margin-top: 10px;
    }


    /* =========================================================
       RESPONSIVE
    ========================================================= */

    @media (max-width: 768px) {

        .main .block-container {
            padding-left: 20px;
            padding-right: 20px;
            padding-top: 28px;
        }

        h1 {
            font-size: 2.75rem !important;
        }

        .description {
            font-size: 0.92rem;
        }

        .brand-meta {
            display: none;
        }
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# MODEL
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

def restore_frame(
    tensor,
    width,
    height
):

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
    """
    <div class="brand-bar">

        <div class="brand">
            NEURAL<span class="brand-dot">.</span>LAB
        </div>

        <div class="brand-meta">
            Computer Vision · Deep Learning
        </div>

    </div>
    """,
    unsafe_allow_html=True
)


st.markdown(
    '<div class="eyebrow">RESEARCH PROJECT · VIDEO COMPRESSION</div>',
    unsafe_allow_html=True
)


st.title("Neural Video Compression")


st.markdown(
    """
    <div class="description">
        A learned video reconstruction system that uses neural
        motion estimation, residual representation, and latent
        feature compression to reconstruct video frames.
    </div>
    """,
    unsafe_allow_html=True
)


st.markdown(
    '<div class="hero-rule"></div>',
    unsafe_allow_html=True
)


# ============================================================
# INPUT
# ============================================================

st.markdown(
    """
    <div class="section-header">
        <span class="section-number">01</span>
        <span class="section-title">Input Video</span>
    </div>
    """,
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
            MP4 · WEBM · AVI · MOV
            <br>
            Upload a video to run the trained neural model.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.stop()


# ============================================================
# SAVE INPUT
# ============================================================

suffix = os.path.splitext(
    uploaded_file.name
)[1]


input_file = tempfile.NamedTemporaryFile(
    delete=False,
    suffix=suffix
)


input_file.write(
    uploaded_file.getvalue()
)

input_file.close()

input_path = input_file.name


# ============================================================
# READ VIDEO INFORMATION
# ============================================================

cap = cv2.VideoCapture(
    input_path
)


if not cap.isOpened():

    st.error(
        "Could not open the uploaded video."
    )

    st.stop()


fps = cap.get(
    cv2.CAP_PROP_FPS
)


if fps <= 0:
    fps = 30.0


width = int(
    cap.get(cv2.CAP_PROP_FRAME_WIDTH)
)


height = int(
    cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
)


frame_count = int(
    cap.get(cv2.CAP_PROP_FRAME_COUNT)
)


cap.release()


# ============================================================
# ORIGINAL VIDEO
# ============================================================

st.markdown(
    '<div class="video-label">Original Video</div>',
    unsafe_allow_html=True
)


st.video(
    uploaded_file.getvalue()
)


# ============================================================
# VIDEO INFORMATION
# ============================================================

st.markdown(
    """
    <div class="section-header">
        <span class="section-number">02</span>
        <span class="section-title">Video Information</span>
    </div>
    """,
    unsafe_allow_html=True
)


c1, c2, c3 = st.columns(3)


with c1:

    st.metric(
        "Resolution",
        f"{width} × {height}"
    )


with c2:

    st.metric(
        "Frame Rate",
        f"{fps:.2f} FPS"
    )


with c3:

    st.metric(
        "Frames",
        frame_count
    )


st.write("")


# ============================================================
# MODEL
# ============================================================

st.markdown(
    """
    <div class="section-header">
        <span class="section-number">03</span>
        <span class="section-title">Model</span>
    </div>
    """,
    unsafe_allow_html=True
)


m1, m2, m3 = st.columns(3)


with m1:

    st.metric(
        "Architecture",
        "CNN + Autoencoder"
    )


with m2:

    st.metric(
        "Model Resolution",
        "448 × 256"
    )


with m3:

    st.metric(
        "Weights",
        "From Scratch"
    )


st.write("")


# ============================================================
# RUN
# ============================================================

run = st.button(
    "Run Neural Compression"
)


if not run:
    st.stop()


# ============================================================
# LOAD MODEL
# ============================================================

with st.spinner(
    "Loading trained model..."
):

    try:

        model = load_model()

    except Exception as error:

        st.error(
            f"Model loading failed: {error}"
        )

        st.stop()


# ============================================================
# OUTPUT FILE
# ============================================================

output_file = tempfile.NamedTemporaryFile(
    delete=False,
    suffix=".mp4"
)


output_file.close()

output_path = output_file.name


# ============================================================
# OPEN VIDEO
# ============================================================

cap = cv2.VideoCapture(
    input_path
)


if not cap.isOpened():

    st.error(
        "Could not reopen the uploaded video."
    )

    st.stop()


writer = cv2.VideoWriter(
    output_path,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)


# ============================================================
# FIRST FRAME
# ============================================================

success, frame = cap.read()


if not success:

    cap.release()
    writer.release()

    st.error(
        "The uploaded video contains no readable frames."
    )

    st.stop()


writer.write(frame)


previous_tensor = preprocess_frame(
    frame
)


processed_frames = 1


# ============================================================
# PROCESSING
# ============================================================

st.markdown(
    """
    <div class="section-header">
        <span class="section-number">04</span>
        <span class="section-title">Processing</span>
    </div>
    """,
    unsafe_allow_html=True
)


progress = st.progress(0)

progress_text = st.empty()


with torch.no_grad():

    while True:

        success, frame = cap.read()

        if not success:
            break


        current_tensor = preprocess_frame(
            frame
        )


        outputs = model(
            previous_tensor,
            current_tensor
        )


        reconstructed = restore_frame(
            outputs["reconstructed_frame"],
            width,
            height
        )


        writer.write(
            reconstructed
        )


        previous_tensor = current_tensor

        processed_frames += 1


        progress_value = min(
            processed_frames /
            max(frame_count, 1),
            1.0
        )


        progress.progress(
            progress_value
        )


        progress_text.caption(
            f"{processed_frames:,} / "
            f"{frame_count:,} frames"
        )


cap.release()
writer.release()


# ============================================================
# RESULT
# ============================================================

st.success(
    f"Processing complete · "
    f"{processed_frames:,} frames reconstructed."
)


# ============================================================
# RECONSTRUCTED VIDEO
# ============================================================

st.markdown(
    """
    <div class="section-header">
        <span class="section-number">05</span>
        <span class="section-title">Reconstructed Video</span>
    </div>
    """,
    unsafe_allow_html=True
)


st.video(
    output_path
)


# ============================================================
# OUTPUT METRICS
# ============================================================

c1, c2, c3 = st.columns(3)


with c1:

    st.metric(
        "Resolution",
        f"{width} × {height}"
    )


with c2:

    st.metric(
        "Frames",
        processed_frames
    )


with c3:

    st.metric(
        "Frame Rate",
        f"{fps:.2f} FPS"
    )


st.write("")


# ============================================================
# DOWNLOAD
# ============================================================

with open(
    output_path,
    "rb"
) as file:

    st.download_button(
        "Download Reconstructed Video",
        data=file,
        file_name="reconstructed_video.mp4",
        mime="video/mp4"
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()


st.markdown(
    """
    <div class="fine-print">
        Neural video reconstruction · inference only ·
        trained without pretrained weights.
    </div>
    """,
    unsafe_allow_html=True
)