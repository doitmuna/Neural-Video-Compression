import os
import sys
import shutil
import subprocess
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
# EXACT EDITORIAL STYLE
# ============================================================

st.markdown(
    """
    <style>

    /* ========================================================
       GLOBAL
    ======================================================== */

    :root {
        --black: #111111;
        --white: #ffffff;
        --gray: #666666;
        --light-gray: #eeeeee;
        --yellow: #ffe500;
    }

    html,
    body,
    [class*="css"],
    [data-testid="stAppViewContainer"] {
        font-family:
            Arial,
            Helvetica,
            sans-serif !important;
    }

    [data-testid="stAppViewContainer"],
    [data-testid="stMain"] {
        background: #ffffff !important;
    }

    .main .block-container {
        max-width: 1160px !important;
        padding-top: 45px !important;
        padding-bottom: 70px !important;
    }

    /* ========================================================
       STREAMLIT HEADER
    ======================================================== */

    [data-testid="stHeader"] {
        background: #ffffff !important;
        box-shadow: none !important;
        border-bottom: 1px solid #111111 !important;
    }

    [data-testid="stDecoration"] {
        display: none !important;
    }

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    /* ========================================================
       HERO
    ======================================================== */

    .eyebrow {
        display: inline-block;
        background: var(--black);
        color: var(--yellow);
        padding: 7px 13px 6px 13px;
        font-family:
            "Courier New",
            Courier,
            monospace !important;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.14em;
        line-height: 1;
        margin-bottom: 28px;
    }

    .hero-title {
        font-family:
            Arial Narrow,
            "Helvetica Neue",
            Arial,
            sans-serif !important;
        color: var(--black);
        font-size: 4rem;
        font-weight: 900;
        letter-spacing: -0.065em;
        line-height: 0.95;
        text-transform: uppercase;
        margin: 0;
    }

    .hero-description {
        color: #333333;
        font-family:
            Arial,
            Helvetica,
            sans-serif !important;
        font-size: 1rem;
        line-height: 1.65;
        max-width: 760px;
        margin-top: 28px;
    }

    .hero-rule {
        height: 6px;
        margin: 42px 0 62px 0;
        background:
            linear-gradient(
                to right,
                var(--yellow) 0%,
                var(--yellow) 10%,
                var(--black) 10%,
                var(--black) 100%
            );
    }

    /* ========================================================
       SECTION HEADERS
    ======================================================== */

    .section-heading {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-top: 32px;
        margin-bottom: 18px;
    }

    .section-marker {
        display: inline-block;
        background: var(--black);
        color: var(--yellow);
        padding: 3px 6px;
        font-family:
            "Courier New",
            Courier,
            monospace !important;
        font-size: 0.8rem;
        font-weight: 700;
    }

    .section-title {
        font-family:
            "Courier New",
            Courier,
            monospace !important;
        color: var(--black);
        font-size: 0.9rem;
        font-weight: 700;
        letter-spacing: 0.11em;
        text-transform: uppercase;
    }

    /* ========================================================
       UPLOADER
    ======================================================== */

    [data-testid="stFileUploader"] {
        background: #fafafa !important;
        border: 2px solid var(--black) !important;
        border-radius: 0 !important;
        padding: 18px !important;
    }

    [data-testid="stFileUploaderDropzone"] {
        min-height: 85px !important;
        background: #ffffff !important;
        border: 1px dashed #222222 !important;
        border-radius: 0 !important;
    }

    [data-testid="stFileUploaderDropzoneInstructions"] {
        color: var(--black) !important;
    }

    [data-testid="stFileUploaderDropzoneInstructions"] span {
        color: var(--black) !important;
    }

    [data-testid="stFileUploaderDropzoneInstructions"] small {
        color: #444444 !important;
    }

    [data-testid="stFileUploaderDropzone"] button {
        background: var(--yellow) !important;
        color: var(--black) !important;
        border: 2px solid var(--black) !important;
        border-radius: 0 !important;
        font-family:
            "Courier New",
            Courier,
            monospace !important;
        font-weight: 700 !important;
    }

    /* ========================================================
       NOTE
    ======================================================== */

    .upload-note {
        border-left: 4px solid var(--yellow);
        padding: 12px 15px;
        margin-top: 18px;
        color: var(--black);
        font-family:
            "Courier New",
            Courier,
            monospace !important;
        font-size: 0.76rem;
        font-weight: 700;
        line-height: 1.5;
        text-transform: uppercase;
    }

    /* ========================================================
       VIDEO LABEL
    ======================================================== */

    .video-label {
        color: #555555;
        font-family:
            "Courier New",
            Courier,
            monospace !important;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin: 20px 0 8px 0;
    }

    /* ========================================================
       METRICS
    ======================================================== */

    [data-testid="stMetric"] {
        background: #ffffff !important;
        border-top: 2px solid var(--black) !important;
        border-bottom: 1px solid #cccccc !important;
        border-left: none !important;
        border-right: none !important;
        border-radius: 0 !important;
        padding: 13px 0 !important;
    }

    [data-testid="stMetricLabel"] {
        color: #666666 !important;
        font-family:
            "Courier New",
            Courier,
            monospace !important;
        font-size: 0.68rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.08em !important;
        text-transform: uppercase !important;
    }

    [data-testid="stMetricValue"] {
        color: var(--black) !important;
        font-family:
            "Courier New",
            Courier,
            monospace !important;
        font-size: 1.25rem !important;
        font-weight: 700 !important;
    }

    /* ========================================================
       BUTTONS
    ======================================================== */

    .stButton > button {
        width: 100%;
        min-height: 50px;
        border-radius: 0 !important;
        border: 2px solid var(--black) !important;
        background: var(--black) !important;
        color: var(--white) !important;
        font-family:
            "Courier New",
            Courier,
            monospace !important;
        font-size: 0.78rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.08em !important;
        text-transform: uppercase !important;
    }

    .stButton > button:hover {
        background: var(--yellow) !important;
        border-color: var(--black) !important;
        color: var(--black) !important;
    }

    .stDownloadButton > button {
        width: 100%;
        min-height: 48px;
        border-radius: 0 !important;
        border: 2px solid var(--black) !important;
        background: var(--white) !important;
        color: var(--black) !important;
        font-family:
            "Courier New",
            Courier,
            monospace !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
    }

    .stDownloadButton > button:hover {
        background: var(--black) !important;
        color: var(--white) !important;
    }

    /* ========================================================
       ALERTS
    ======================================================== */

    [data-testid="stAlert"] {
        border-radius: 0 !important;
    }

    /* ========================================================
       DIVIDERS
    ======================================================== */

    hr {
        border: none !important;
        border-top: 1px solid #cccccc !important;
        margin: 35px 0 !important;
    }

    /* ========================================================
       SMALL TEXT
    ======================================================== */

    .small-note {
        color: #666666;
        font-family:
            "Courier New",
            Courier,
            monospace !important;
        font-size: 0.72rem;
        line-height: 1.55;
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
            f"Checkpoint not found:\n{MODEL_PATH}"
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
# FRAME PREPROCESSING
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
# FRAME RESTORATION
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
# FFMPEG
# ============================================================

def get_ffmpeg_path():

    path = shutil.which("ffmpeg")

    if path is None:
        raise RuntimeError(
            "FFmpeg is not available."
        )

    return path


def encode_browser_video(
    input_path,
    output_path
):

    ffmpeg = get_ffmpeg_path()

    command = [
        ffmpeg,
        "-y",
        "-i",
        input_path,
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        output_path
    ]

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    if result.returncode != 0:

        raise RuntimeError(
            result.stderr[-4000:]
        )


# ============================================================
# HERO
# ============================================================

st.markdown(
    '<div class="eyebrow">DEEP LEARNING // COMPUTER VISION</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="hero-title">NEURAL VIDEO COMPRESSION</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="hero-description">
        A learned frame-reconstruction system utilizing neural
        motion estimation, residual coding, and latent
        representation.
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
    <div class="section-heading">
        <span class="section-marker">//</span>
        <span class="section-title">01 / INPUT VIDEO</span>
    </div>
    """,
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Upload",
    type=["mp4", "webm", "avi", "mov"],
    label_visibility="collapsed"
)


if uploaded_file is None:

    st.markdown(
        """
        <div class="upload-note">
            Upload a video file to run frame reconstruction
            through the trained neural model.
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

input_temp = tempfile.NamedTemporaryFile(
    delete=False,
    suffix=suffix
)

input_temp.write(
    uploaded_file.getvalue()
)

input_temp.close()

input_path = input_temp.name


# ============================================================
# VIDEO INFORMATION
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
# ORIGINAL
# ============================================================

st.markdown(
    '<div class="video-label">ORIGINAL VIDEO</div>',
    unsafe_allow_html=True
)

st.video(
    uploaded_file.getvalue()
)


# ============================================================
# VIDEO INFO
# ============================================================

st.markdown(
    '<div class="video-label">VIDEO INFORMATION</div>',
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
        "FPS",
        f"{fps:.2f}"
    )

with c3:
    st.metric(
        "Frames",
        f"{frame_count:,}"
    )


# ============================================================
# MODEL INFO
# ============================================================

st.markdown(
    '<div class="video-label">MODEL</div>',
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
        "Input",
        "448 × 256"
    )

with m3:
    st.metric(
        "Training",
        "From Scratch"
    )


st.divider()


# ============================================================
# RUN BUTTON
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
            f"Model loading failed:\n{error}"
        )

        st.stop()


# ============================================================
# TEMPORARY FILES
# ============================================================

raw_temp = tempfile.NamedTemporaryFile(
    delete=False,
    suffix=".avi"
)

raw_temp.close()

raw_output_path = raw_temp.name


final_temp = tempfile.NamedTemporaryFile(
    delete=False,
    suffix=".mp4"
)

final_temp.close()

final_output_path = final_temp.name


# ============================================================
# OPEN INPUT
# ============================================================

cap = cv2.VideoCapture(
    input_path
)

writer = cv2.VideoWriter(
    raw_output_path,
    cv2.VideoWriter_fourcc(*"MJPG"),
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
# PROCESS
# ============================================================

st.markdown(
    '<div class="section-heading"><span class="section-marker">//</span><span class="section-title">02 / NEURAL RECONSTRUCTION</span></div>',
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

        progress.progress(
            min(
                processed_frames
                / max(frame_count, 1),
                1.0
            )
        )

        progress_text.caption(
            f"{processed_frames:,} / "
            f"{frame_count:,} frames"
        )


cap.release()
writer.release()


# ============================================================
# WEB ENCODING
# ============================================================

st.markdown(
    '<div class="section-heading"><span class="section-marker">//</span><span class="section-title">03 / WEB ENCODING</span></div>',
    unsafe_allow_html=True
)

with st.spinner(
    "Encoding browser-compatible MP4..."
):

    try:

        encode_browser_video(
            raw_output_path,
            final_output_path
        )

    except Exception as error:

        st.error(
            f"Encoding failed:\n{error}"
        )

        st.stop()


# ============================================================
# FINAL VIDEO
# ============================================================

with open(
    final_output_path,
    "rb"
) as file:

    final_video = file.read()


st.success(
    f"Processing complete · "
    f"{processed_frames:,} frames reconstructed."
)


st.markdown(
    '<div class="section-heading"><span class="section-marker">//</span><span class="section-title">04 / RECONSTRUCTED VIDEO</span></div>',
    unsafe_allow_html=True
)

st.video(
    final_video
)


# ============================================================
# OUTPUT INFO
# ============================================================

o1, o2, o3 = st.columns(3)

with o1:
    st.metric(
        "Resolution",
        f"{width} × {height}"
    )

with o2:
    st.metric(
        "FPS",
        f"{fps:.2f}"
    )

with o3:
    st.metric(
        "Frames",
        f"{processed_frames:,}"
    )


# ============================================================
# DOWNLOAD
# ============================================================

st.write("")

st.download_button(
    label="Download Reconstructed Video",
    data=final_video,
    file_name="reconstructed_video.mp4",
    mime="video/mp4"
)


st.divider()

st.markdown(
    """
    <div class="small-note">
        OUTPUT: H.264 MP4 · BROWSER COMPATIBLE<br>
        LEARNED BPP IS A MODEL RATE ESTIMATE, NOT THE FINAL MP4 BITRATE.
    </div>
    """,
    unsafe_allow_html=True
)