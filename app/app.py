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

APP_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

sys.path.insert(0, APP_DIR)

from neural_codec import NeuralVideoCompressionModel


MODEL_PATH = os.path.abspath(
    os.path.join(
        APP_DIR,
        "..",
        "checkpoints",
        "best_model.pth"
    )
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
    page_icon="▣",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# CLEAN WHITE STYLE
# ============================================================

st.markdown(
    """
    <style>

    /* Main page */

    .stApp {
        background: #ffffff;
        color: #111111;
    }

    .block-container {
        max-width: 1100px;
        padding-top: 45px;
        padding-bottom: 50px;
    }

    /* Typography */

    html,
    body,
    [class*="css"] {
        font-family:
            ui-monospace,
            SFMono-Regular,
            Menlo,
            Monaco,
            Consolas,
            "Liberation Mono",
            monospace;
    }

    h1 {
        font-family:
            ui-monospace,
            SFMono-Regular,
            Menlo,
            Monaco,
            Consolas,
            "Liberation Mono",
            monospace !important;

        font-size: 2.2rem !important;
        font-weight: 700 !important;
        letter-spacing: -0.04em !important;
        color: #111111 !important;
    }

    p {
        color: #555555;
    }

    /* Hide Streamlit branding */

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    header {
        visibility: hidden;
    }

    /* Buttons */

    .stButton > button {
        width: 100%;
        border: 1px solid #111111;
        border-radius: 4px;
        background: #111111;
        color: #ffffff;
        font-family: inherit;
        font-weight: 700;
        padding: 0.65rem 1rem;
    }

    .stButton > button:hover {
        border-color: #333333;
        background: #333333;
        color: #ffffff;
    }

    .stDownloadButton > button {
        width: 100%;
        border-radius: 4px;
        font-family: inherit;
        font-weight: 600;
    }

    /* File uploader */

    [data-testid="stFileUploader"] {
        border: 1px solid #dddddd;
        border-radius: 6px;
        background: #fafafa;
        padding: 8px;
    }

    /* Metrics */

    [data-testid="stMetric"] {
        background: #fafafa;
        border: 1px solid #e5e5e5;
        border-radius: 5px;
        padding: 12px;
    }

    [data-testid="stMetricLabel"] {
        font-family: inherit;
        color: #777777;
    }

    [data-testid="stMetricValue"] {
        font-family: inherit;
        color: #111111;
    }

    /* Divider */

    hr {
        border: none;
        border-top: 1px solid #e5e5e5;
        margin: 28px 0;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD MODEL
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
# PREPROCESS FRAME
# ============================================================

def preprocess_frame(frame):

    frame_rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    frame_rgb = cv2.resize(
        frame_rgb,
        (
            MODEL_WIDTH,
            MODEL_HEIGHT
        ),
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
# RESTORE FRAME
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

st.title("Neural Video Compression")

st.caption(
    "Learned motion estimation + residual coding "
    "trained entirely from scratch."
)

st.caption(
    f"Device: {DEVICE.type.upper()}  |  "
    "Model: CNN + Residual Autoencoder"
)

st.divider()


# ============================================================
# UPLOAD
# ============================================================

st.subheader("Input Video")

uploaded_file = st.file_uploader(
    "Upload a video",
    type=[
        "mp4",
        "webm",
        "avi",
        "mov"
    ]
)


if uploaded_file is None:

    st.info(
        "Upload a video to start neural reconstruction."
    )

    st.stop()


# ============================================================
# SAVE INPUT TEMPORARILY
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
    cap.get(
        cv2.CAP_PROP_FRAME_WIDTH
    )
)

height = int(
    cap.get(
        cv2.CAP_PROP_FRAME_HEIGHT
    )
)

metadata_frame_count = int(
    cap.get(
        cv2.CAP_PROP_FRAME_COUNT
    )
)

cap.release()


# ============================================================
# ORIGINAL VIDEO
# ============================================================

st.subheader("Original")

st.video(
    uploaded_file.getvalue()
)


# ============================================================
# VIDEO INFORMATION
# ============================================================

st.subheader("Video Information")

info1, info2, info3 = st.columns(3)

with info1:
    st.metric(
        "Resolution",
        f"{width} × {height}"
    )

with info2:
    st.metric(
        "Frame Rate",
        f"{fps:.2f} FPS"
    )

with info3:
    st.metric(
        "Frames",
        metadata_frame_count
    )


# ============================================================
# MODEL INFORMATION
# ============================================================

st.subheader("Model")

model1, model2, model3 = st.columns(3)

with model1:
    st.metric(
        "Architecture",
        "CNN + AE"
    )

with model2:
    st.metric(
        "Internal Resolution",
        "448 × 256"
    )

with model3:
    st.metric(
        "Training",
        "From Scratch"
    )


st.divider()


# ============================================================
# RUN BUTTON
# ============================================================

run_compression = st.button(
    "RUN NEURAL COMPRESSION"
)

if not run_compression:
    st.stop()


# ============================================================
# LOAD MODEL
# ============================================================

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

output_temp = tempfile.NamedTemporaryFile(
    delete=False,
    suffix=".mp4"
)

output_temp.close()

output_path = output_temp.name


# ============================================================
# OPEN INPUT
# ============================================================

cap = cv2.VideoCapture(
    input_path
)

if not cap.isOpened():

    st.error(
        "Could not reopen uploaded video."
    )

    st.stop()


writer = cv2.VideoWriter(
    output_path,
    cv2.VideoWriter_fourcc(
        *"mp4v"
    ),
    fps,
    (
        width,
        height
    )
)


# ============================================================
# READ FIRST FRAME
# ============================================================

success, frame = cap.read()

if not success:

    cap.release()
    writer.release()

    st.error(
        "The video contains no readable frames."
    )

    st.stop()


# Keep first frame as reference
writer.write(frame)

previous_tensor = preprocess_frame(
    frame
)

processed_frames = 1


# ============================================================
# PROCESS VIDEO
# ============================================================

st.subheader("Processing")

progress = st.progress(0)

progress_text = st.empty()


with torch.no_grad():

    while True:

        success, frame = cap.read()

        if not success:
            break

        current_tensor = (
            preprocess_frame(frame)
        )

        outputs = model(
            previous_tensor,
            current_tensor
        )

        reconstructed = restore_frame(
            outputs[
                "reconstructed_frame"
            ],
            width,
            height
        )

        writer.write(
            reconstructed
        )

        previous_tensor = (
            current_tensor
        )

        processed_frames += 1

        progress_value = min(
            processed_frames
            / max(
                metadata_frame_count,
                1
            ),
            1.0
        )

        progress.progress(
            progress_value
        )

        progress_text.write(
            f"Processing frame "
            f"{processed_frames} / "
            f"{metadata_frame_count}"
        )


cap.release()
writer.release()


# ============================================================
# COMPLETION
# ============================================================

st.success(
    f"Compression complete — "
    f"{processed_frames} frames processed."
)


# ============================================================
# OUTPUT
# ============================================================

st.subheader("Reconstructed Video")

st.video(
    output_path
)


# ============================================================
# OUTPUT INFORMATION
# ============================================================

out1, out2, out3 = st.columns(3)

with out1:
    st.metric(
        "Resolution",
        f"{width} × {height}"
    )

with out2:
    st.metric(
        "Frames",
        processed_frames
    )

with out3:
    st.metric(
        "FPS",
        f"{fps:.2f}"
    )


# ============================================================
# DOWNLOAD
# ============================================================

st.write("")

with open(
    output_path,
    "rb"
) as file:

    st.download_button(
        label="DOWNLOAD RECONSTRUCTED VIDEO",
        data=file,
        file_name="reconstructed_video.mp4",
        mime="video/mp4"
    )


st.divider()

st.caption(
    "Inference only. No pretrained model or "
    "pretrained weights are used."
)