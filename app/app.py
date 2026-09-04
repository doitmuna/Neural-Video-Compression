import os
import sys
import tempfile

import cv2
import numpy as np
import streamlit as st
import torch
import imageio_ffmpeg


# ============================================================
# PATHS
# ============================================================

APP_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

PROJECT_ROOT = os.path.abspath(
    os.path.join(APP_DIR, "..")
)

sys.path.insert(0, APP_DIR)

from neural_codec import NeuralVideoCompressionModel


MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "checkpoints",
    "best_model.pth"
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

MODEL_WIDTH = 448
MODEL_HEIGHT = 256


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Neural Video Compression",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# LIGHT EDITORIAL UI
# ============================================================

st.markdown(
    """
    <style>

    :root {
        --black: #111111;
        --white: #ffffff;
        --gray: #666666;
        --light: #f7f7f7;
        --line: #d6d6d6;
        --yellow: #ffe500;
    }

    /* ---------- Page ---------- */

    html,
    body {
        background: #ffffff !important;
    }

    [data-testid="stAppViewContainer"],
    [data-testid="stMain"] {
        background: #ffffff !important;
    }

    .block-container {
        max-width: 1160px !important;
        padding-top: 40px !important;
        padding-bottom: 60px !important;
    }

    /* ---------- Font ---------- */

    html,
    body,
    [class*="css"] {
        font-family:
            Arial,
            Helvetica,
            sans-serif !important;
    }

    /* ---------- Streamlit header ---------- */

    [data-testid="stHeader"] {
        background: #ffffff !important;
        box-shadow: none !important;
        border-bottom: 1px solid #dddddd !important;
    }

    [data-testid="stDecoration"] {
        display: none !important;
    }

    /* ---------- Titles ---------- */

    .eyebrow {
        display: inline-block;
        background: #111111;
        color: #ffe500;
        padding: 7px 12px;
        font-family:
            "Courier New",
            Courier,
            monospace !important;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.13em;
        margin-bottom: 22px;
    }

    .hero-title {
        color: #111111;
        font-family:
            "Arial Narrow",
            "Helvetica Neue",
            Arial,
            sans-serif !important;
        font-size: 3.8rem;
        font-weight: 900;
        letter-spacing: -0.06em;
        line-height: 0.95;
        text-transform: uppercase;
        margin: 0;
    }

    .hero-description {
        max-width: 780px;
        color: #333333;
        font-size: 1rem;
        line-height: 1.6;
        margin-top: 24px;
    }

    .hero-rule {
        height: 6px;
        margin: 38px 0 58px 0;
        background:
            linear-gradient(
                to right,
                #ffe500 0%,
                #ffe500 10%,
                #111111 10%,
                #111111 100%
            );
    }

    /* ---------- Section headers ---------- */

    .section-heading {
        display: flex;
        align-items: center;
        gap: 9px;
        margin-top: 28px;
        margin-bottom: 17px;
    }

    .section-marker {
        background: #111111;
        color: #ffe500;
        padding: 3px 6px;
        font-family:
            "Courier New",
            Courier,
            monospace !important;
        font-size: 0.78rem;
        font-weight: 700;
    }

    .section-title {
        color: #111111;
        font-family:
            "Courier New",
            Courier,
            monospace !important;
        font-size: 0.83rem;
        font-weight: 700;
        letter-spacing: 0.1em;
        text-transform: uppercase;
    }

    /* ---------- File uploader ---------- */

    [data-testid="stFileUploader"] {
        background: #fafafa !important;
        border: 2px solid #111111 !important;
        border-radius: 0 !important;
        padding: 16px !important;
    }

    [data-testid="stFileUploaderDropzone"] {
        background: #ffffff !important;
        border: 1px dashed #333333 !important;
        border-radius: 0 !important;
    }

    [data-testid="stFileUploaderDropzoneInstructions"] {
        color: #222222 !important;
    }

    [data-testid="stFileUploaderDropzoneInstructions"] span,
    [data-testid="stFileUploaderDropzoneInstructions"] small {
        color: #333333 !important;
    }

    [data-testid="stFileUploaderDropzone"] button {
        background: #ffe500 !important;
        color: #111111 !important;
        border: 2px solid #111111 !important;
        border-radius: 0 !important;
        font-family:
            "Courier New",
            Courier,
            monospace !important;
        font-weight: 700 !important;
    }

    /* ---------- Metrics ---------- */

    [data-testid="stMetric"] {
        background: #ffffff !important;
        border-top: 2px solid #111111 !important;
        border-bottom: 1px solid #cccccc !important;
        border-left: none !important;
        border-right: none !important;
        border-radius: 0 !important;
        padding: 12px 0 !important;
    }

    [data-testid="stMetricLabel"] {
        color: #666666 !important;
        font-family:
            "Courier New",
            Courier,
            monospace !important;
        font-size: 0.67rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.08em !important;
        text-transform: uppercase !important;
    }

    [data-testid="stMetricValue"] {
        color: #111111 !important;
        font-family:
            "Courier New",
            Courier,
            monospace !important;
        font-size: 1.2rem !important;
        font-weight: 700 !important;
    }

    /* ---------- Buttons ---------- */

    .stButton > button {
        width: 100%;
        min-height: 48px;
        background: #111111 !important;
        color: #ffffff !important;
        border: 2px solid #111111 !important;
        border-radius: 0 !important;
        font-family:
            "Courier New",
            Courier,
            monospace !important;
        font-size: 0.78rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.07em !important;
        text-transform: uppercase !important;
    }

    .stButton > button:hover {
        background: #ffe500 !important;
        color: #111111 !important;
        border-color: #111111 !important;
    }

    .stDownloadButton > button {
        width: 100%;
        min-height: 46px;
        background: #ffffff !important;
        color: #111111 !important;
        border: 2px solid #111111 !important;
        border-radius: 0 !important;
        font-family:
            "Courier New",
            Courier,
            monospace !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
    }

    .stDownloadButton > button:hover {
        background: #111111 !important;
        color: #ffffff !important;
    }

    /* ---------- Notes ---------- */

    .note {
        color: #666666;
        font-family:
            "Courier New",
            Courier,
            monospace !important;
        font-size: 0.72rem;
        line-height: 1.55;
    }

    /* ---------- Dividers ---------- */

    hr {
        border: none !important;
        border-top: 1px solid #cccccc !important;
        margin: 32px 0 !important;
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
        2,
        0,
        1
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
# H.264 WEB ENCODING
# ============================================================

def encode_for_web(
    reconstructed_video,
    original_video,
    output_video
):

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

    command = [
        ffmpeg,
        "-y",

        # Reconstructed video
        "-i",
        reconstructed_video,

        # Original video - used only for audio
        "-i",
        original_video,

        # Video
        "-map",
        "0:v:0",

        # Optional original audio
        "-map",
        "1:a:0?",

        # Browser-compatible H.264
        "-c:v",
        "libx264",

        "-preset",
        "medium",

        "-crf",
        "23",

        "-pix_fmt",
        "yuv420p",

        # Audio if present
        "-c:a",
        "aac",

        "-b:a",
        "128k",

        # Browser playback optimization
        "-movflags",
        "+faststart",

        output_video
    ]

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    if result.returncode != 0:
        raise RuntimeError(
            "FFmpeg encoding failed:\n\n"
            + result.stderr[-5000:]
        )


# ============================================================
# HEADER
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
        Neural video reconstruction using learned motion
        estimation and residual coding.
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    '<div class="hero-rule"></div>',
    unsafe_allow_html=True
)


# ============================================================
# INPUT VIDEO
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
    "Upload video",
    type=[
        "mp4",
        "webm",
        "avi",
        "mov"
    ],
    label_visibility="collapsed"
)


if uploaded_file is None:

    st.markdown(
        """
        <div class="note">
            Upload a video file to run frame reconstruction
            through the trained neural model.
            <br><br>
            MP4 · WEBM · AVI · MOV
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
# READ VIDEO METADATA
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
    """
    <div class="section-heading">
        <span class="section-marker">//</span>
        <span class="section-title">02 / ORIGINAL VIDEO</span>
    </div>
    """,
    unsafe_allow_html=True
)

st.video(
    uploaded_file.getvalue()
)


# ============================================================
# VIDEO INFORMATION
# ============================================================

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Resolution",
        f"{width} × {height}"
    )

with col2:
    st.metric(
        "FPS",
        f"{fps:.2f}"
    )

with col3:
    st.metric(
        "Frames",
        f"{frame_count:,}"
    )


st.divider()


# ============================================================
# MODEL INFORMATION
# ============================================================

st.markdown(
    """
    <div class="section-heading">
        <span class="section-marker">//</span>
        <span class="section-title">03 / MODEL</span>
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
        "Input",
        "448 × 256"
    )

with m3:
    st.metric(
        "Training",
        "From Scratch"
    )


st.write("")


# ============================================================
# RUN MODEL
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
# TEMPORARY RECONSTRUCTION FILE
# ============================================================

raw_temp = tempfile.NamedTemporaryFile(
    delete=False,
    suffix=".avi"
)

raw_temp.close()

raw_output_path = raw_temp.name


# ============================================================
# FINAL WEB VIDEO
# ============================================================

final_temp = tempfile.NamedTemporaryFile(
    delete=False,
    suffix=".mp4"
)

final_temp.close()

final_output_path = final_temp.name


# ============================================================
# OPEN VIDEO
# ============================================================

cap = cv2.VideoCapture(
    input_path
)

writer = cv2.VideoWriter(
    raw_output_path,
    cv2.VideoWriter_fourcc(
        *"MJPG"
    ),
    fps,
    (width, height)
)

if not writer.isOpened():

    cap.release()

    st.error(
        "Could not create reconstruction video."
    )

    st.stop()


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


# Preserve first frame
writer.write(frame)

previous_tensor = preprocess_frame(
    frame
)

processed_frames = 1


# ============================================================
# NEURAL RECONSTRUCTION
# ============================================================

st.markdown(
    """
    <div class="section-heading">
        <span class="section-marker">//</span>
        <span class="section-title">04 / NEURAL RECONSTRUCTION</span>
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

        progress.progress(
            min(
                processed_frames
                / max(frame_count, 1),
                1.0
            )
        )

        progress_text.caption(
            f"Reconstructing frame "
            f"{processed_frames:,} / "
            f"{frame_count:,}"
        )


cap.release()
writer.release()


# ============================================================
# WEB ENCODING
# ============================================================

st.markdown(
    """
    <div class="section-heading">
        <span class="section-marker">//</span>
        <span class="section-title">05 / WEB ENCODING</span>
    </div>
    """,
    unsafe_allow_html=True
)

with st.spinner(
    "Preparing browser-compatible MP4..."
):

    try:

        encode_for_web(
            raw_output_path,
            input_path,
            final_output_path
        )

    except Exception as error:

        st.error(
            str(error)
        )

        st.stop()


# ============================================================
# READ FINAL VIDEO
# ============================================================

with open(
    final_output_path,
    "rb"
) as file:

    final_video_bytes = file.read()


# ============================================================
# COMPLETE
# ============================================================

st.success(
    f"Processing complete · "
    f"{processed_frames:,} frames reconstructed."
)


# ============================================================
# RECONSTRUCTED VIDEO — PLAY DIRECTLY ON PAGE
# ============================================================

st.markdown(
    """
    <div class="section-heading">
        <span class="section-marker">//</span>
        <span class="section-title">06 / RECONSTRUCTED VIDEO</span>
    </div>
    """,
    unsafe_allow_html=True
)

# This embeds the actual generated MP4 in the webpage.
st.video(
    final_video_bytes
)


# ============================================================
# OUTPUT INFORMATION
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
    data=final_video_bytes,
    file_name="reconstructed_video.mp4",
    mime="video/mp4"
)


# ============================================================
# FOOTNOTE
# ============================================================

st.divider()

st.markdown(
    """
    <div class="note">
        The displayed output is a reconstructed H.264 MP4
        generated from the neural model. The learned BPP
        reported during evaluation is a model rate estimate
        and is separate from the final MP4 container size.
    </div>
    """,
    unsafe_allow_html=True
)