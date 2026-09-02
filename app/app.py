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
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="collapsed"
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
    "Deep learning based video reconstruction"
)

st.divider()


# ============================================================
# UPLOAD
# ============================================================

st.subheader("Upload Video")

uploaded_file = st.file_uploader(
    "Choose a video",
    type=[
        "mp4",
        "webm",
        "avi",
        "mov"
    ]
)

if uploaded_file is None:

    st.info(
        "Upload a video to run the trained model."
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
    cap.get(
        cv2.CAP_PROP_FRAME_WIDTH
    )
)

height = int(
    cap.get(
        cv2.CAP_PROP_FRAME_HEIGHT
    )
)

frame_count = int(
    cap.get(
        cv2.CAP_PROP_FRAME_COUNT
    )
)

cap.release()


# ============================================================
# ORIGINAL VIDEO
# ============================================================

st.subheader("Original Video")

st.video(
    uploaded_file.getvalue()
)


# ============================================================
# VIDEO INFORMATION
# ============================================================

st.subheader("Video Information")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Resolution",
        f"{width} × {height}"
    )

with col2:
    st.metric(
        "Frame Rate",
        f"{fps:.2f} FPS"
    )

with col3:
    st.metric(
        "Frames",
        frame_count
    )


st.divider()


# ============================================================
# RUN MODEL
# ============================================================

run_compression = st.button(
    "Run Neural Compression"
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
                frame_count,
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
            f"{frame_count}"
        )


cap.release()
writer.release()


# ============================================================
# RESULT
# ============================================================

st.success(
    f"Compression complete — "
    f"{processed_frames} frames processed."
)


# ============================================================
# RECONSTRUCTED VIDEO
# ============================================================

st.subheader("Reconstructed Video")

st.video(
    output_path
)


# ============================================================
# OUTPUT INFORMATION
# ============================================================

st.subheader("Output Information")

out1, out2, out3 = st.columns(3)

with out1:
    st.metric(
        "Resolution",
        f"{width} × {height}"
    )

with out2:
    st.metric(
        "Frames Processed",
        processed_frames
    )

with out3:
    st.metric(
        "Frame Rate",
        f"{fps:.2f} FPS"
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
        label="Download Reconstructed Video",
        data=file,
        file_name="reconstructed_video.mp4",
        mime="video/mp4"
    )