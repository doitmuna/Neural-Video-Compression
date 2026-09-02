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
    "cuda" if torch.cuda.is_available() else "cpu"
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
# MINIMAL DARK / MONOSPACE UI
# ============================================================

st.markdown(
    """
    <style>

    /* ---------- Global ---------- */

    html, body, [class*="css"] {
        font-family:
            ui-monospace,
            SFMono-Regular,
            Menlo,
            Monaco,
            Consolas,
            "Liberation Mono",
            monospace;
    }

    .stApp {
        background: #0b0d10;
        color: #e6e8eb;
    }

    .block-container {
        max-width: 1180px;
        padding-top: 3rem;
        padding-bottom: 3rem;
    }

    /* ---------- Hide Streamlit chrome ---------- */

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    header {
        visibility: hidden;
    }

    /* ---------- Typography ---------- */

    .hero {
        border-left: 2px solid #e6e8eb;
        padding-left: 18px;
        margin-bottom: 34px;
    }

    .hero-title {
        font-size: 2.2rem;
        font-weight: 700;
        letter-spacing: -0.04em;
        line-height: 1.1;
    }

    .hero-subtitle {
        margin-top: 9px;
        color: #8b9199;
        font-size: 0.88rem;
        line-height: 1.6;
    }

    .tag {
        display: inline-block;
        margin-top: 12px;
        padding: 4px 8px;
        border: 1px solid #30343a;
        border-radius: 4px;
        color: #aeb4bc;
        font-size: 0.72rem;
        letter-spacing: 0.04em;
    }

    /* ---------- Upload ---------- */

    .upload-label {
        color: #8b9199;
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        margin-bottom: 8px;
    }

    /* ---------- Cards ---------- */

    .panel {
        border: 1px solid #24282e;
        border-radius: 8px;
        background: #0f1216;
        padding: 14px;
        margin-bottom: 18px;
    }

    .panel-title {
        color: #8b9199;
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        margin-bottom: 12px;
    }

    /* ---------- Metrics ---------- */

    .metric-box {
        border-top: 1px solid #24282e;
        padding-top: 10px;
        margin-top: 2px;
    }

    .metric-label {
        color: #747b84;
        font-size: 0.68rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }

    .metric-value {
        margin-top: 4px;
        color: #e6e8eb;
        font-size: 1.15rem;
        font-weight: 600;
    }

    /* ---------- Buttons ---------- */

    .stButton > button {
        width: 100%;
        border-radius: 5px;
        border: 1px solid #3a3f46;
        background: #e6e8eb;
        color: #0b0d10;
        font-family: inherit;
        font-weight: 700;
        letter-spacing: 0.02em;
    }

    .stButton > button:hover {
        border-color: #ffffff;
        background: #ffffff;
        color: #000000;
    }

    /* ---------- Download button ---------- */

    .stDownloadButton > button {
        width: 100%;
        border-radius: 5px;
        font-family: inherit;
        font-weight: 600;
    }

    /* ---------- Success ---------- */

    .status {
        border: 1px solid #26352b;
        background: #101711;
        color: #8fd19a;
        border-radius: 6px;
        padding: 10px 12px;
        margin: 14px 0 20px 0;
        font-size: 0.78rem;
    }

    /* ---------- Small text ---------- */

    .fine {
        color: #676d75;
        font-size: 0.68rem;
        line-height: 1.6;
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
# PREPROCESSING
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

st.markdown(
    """
    <div class="hero">
        <div class="hero-title">
            neural / video compression
        </div>

        <div class="hero-subtitle">
            Learned motion estimation + residual coding
            <br>
            trained entirely from scratch
        </div>

        <div class="tag">
            PYTORCH · CNN · RATE–DISTORTION
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# MODEL STATUS
# ============================================================

status_col1, status_col2 = st.columns(
    [3, 1]
)

with status_col1:

    st.markdown(
        """
        <div class="upload-label">
            input
        </div>
        """,
        unsafe_allow_html=True
    )

with status_col2:

    device_name = (
        "CUDA"
        if torch.cuda.is_available()
        else "CPU"
    )

    st.markdown(
        f"""
        <div style="
            text-align:right;
            color:#676d75;
            font-size:0.68rem;
            padding-top:3px;
        ">
            device: {device_name}
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
        <div class="panel">

            <div class="panel-title">
                waiting for video
            </div>

            <div class="fine">
                Supported formats:
                MP4 · WEBM · AVI · MOV
                <br><br>
                The uploaded video is processed frame-by-frame
                using the trained neural compression model.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.stop()


# ============================================================
# SAVE UPLOAD
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
# INPUT PREVIEW
# ============================================================

left, right = st.columns(
    2,
    gap="medium"
)

with left:

    st.markdown(
        """
        <div class="panel-title">
            original
        </div>
        """,
        unsafe_allow_html=True
    )

    st.video(
        uploaded_file.getvalue()
    )


with right:

    st.markdown(
        """
        <div class="panel-title">
            configuration
        </div>
        """,
        unsafe_allow_html=True
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(
            f"""
            <div class="metric-box">
                <div class="metric-label">
                    resolution
                </div>
                <div class="metric-value">
                    {width}×{height}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:
        st.markdown(
            f"""
            <div class="metric-box">
                <div class="metric-label">
                    fps
                </div>
                <div class="metric-value">
                    {fps:.0f}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c3:
        st.markdown(
            f"""
            <div class="metric-box">
                <div class="metric-label">
                    frames
                </div>
                <div class="metric-value">
                    {metadata_frame_count}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.write("")

    st.markdown(
        """
        <div class="fine">
            Internal model resolution:
            <b>448×256</b>
            <br>
            Motion is estimated between consecutive
            frames and the remaining residual is
            reconstructed by the learned autoencoder.
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# COMPRESS BUTTON
# ============================================================

st.write("")

compress = st.button(
    "RUN NEURAL COMPRESSION"
)

if not compress:
    st.markdown(
        """
        <div class="fine" style="margin-top:10px;">
            Press the button to start inference.
        </div>
        """,
        unsafe_allow_html=True
    )

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
# PROCESS VIDEO
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
    cv2.VideoWriter_fourcc(
        *"mp4v"
    ),
    fps,
    (
        width,
        height
    )
)


success, frame = cap.read()

if not success:

    cap.release()
    writer.release()

    st.error(
        "The uploaded video contains no readable frames."
    )

    st.stop()


# First frame is kept as reference
writer.write(
    frame
)

previous_tensor = preprocess_frame(
    frame
)

processed_frames = 1

progress = st.progress(
    0
)

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

        progress_text.caption(
            f"processing frame "
            f"{processed_frames}"
        )


cap.release()
writer.release()


# ============================================================
# RESULT
# ============================================================

st.markdown(
    f"""
    <div class="status">
        ✓ compression complete ·
        {processed_frames} frames processed
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# OUTPUT PREVIEW
# ============================================================

st.markdown(
    """
    <div class="panel-title">
        reconstructed output
    </div>
    """,
    unsafe_allow_html=True
)

st.video(
    output_path
)


# ============================================================
# RESULT METRICS
# ============================================================

st.write("")

m1, m2, m3 = st.columns(3)

with m1:

    st.markdown(
        f"""
        <div class="metric-box">
            <div class="metric-label">
                resolution
            </div>
            <div class="metric-value">
                {width}×{height}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with m2:

    st.markdown(
        f"""
        <div class="metric-box">
            <div class="metric-label">
                frames processed
            </div>
            <div class="metric-value">
                {processed_frames}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with m3:

    st.markdown(
        f"""
        <div class="metric-box">
            <div class="metric-label">
                model input
            </div>
            <div class="metric-value">
                {MODEL_WIDTH}×{MODEL_HEIGHT}
            </div>
        </div>
        """,
        unsafe_allow_html=True
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


st.markdown(
    """
    <div class="fine" style="margin-top:20px;">
        This application performs inference only.
        The neural model was trained from scratch;
        no pretrained weights are used.
    </div>
    """,
    unsafe_allow_html=True
)