
import os
import sys
import tempfile

import cv2
import numpy as np
import streamlit as st
import torch


# Allow import from this directory
APP_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

sys.path.insert(
    0,
    APP_DIR
)

from neural_codec import NeuralVideoCompressionModel


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

MODEL_PATH = os.path.abspath(
    os.path.join(
        APP_DIR,
        "..",
        "checkpoints",
        "best_model.pth"
    )
)

MODEL_WIDTH = 448
MODEL_HEIGHT = 256


# ------------------------------------------------------------
# Load trained model
# ------------------------------------------------------------

@st.cache_resource
def load_model():

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Checkpoint not found: {MODEL_PATH}"
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


# ------------------------------------------------------------
# Frame preprocessing
# ------------------------------------------------------------

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
        torch.from_numpy(
            frame_rgb
        )
        .float()
        / 255.0
    )

    tensor = tensor.permute(
        2, 0, 1
    ).unsqueeze(0)

    return tensor.to(DEVICE)


# ------------------------------------------------------------
# Frame restoration
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# Streamlit interface
# ------------------------------------------------------------

st.set_page_config(
    page_title="Neural Video Compression",
    page_icon="🎥",
    layout="wide"
)

st.title("Neural Video Compression")

st.write(
    "Video reconstruction using a neural "
    "motion estimation network and residual "
    "autoencoder trained from scratch."
)

st.caption(
    "No pretrained model or pretrained weights are used."
)

uploaded_file = st.file_uploader(
    "Upload a video",
    type=[
        "mp4",
        "webm",
        "avi",
        "mov"
    ]
)


if uploaded_file is not None:

    try:

        model = load_model()

        suffix = os.path.splitext(
            uploaded_file.name
        )[1]

        input_temp = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix
        )

        input_temp.write(
            uploaded_file.read()
        )

        input_temp.close()

        input_path = input_temp.name

        output_temp = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp4"
        )

        output_temp.close()

        output_path = output_temp.name

        cap = cv2.VideoCapture(
            input_path
        )

        if not cap.isOpened():
            raise RuntimeError(
                "Could not open uploaded video."
            )

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

        writer = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(
                *"mp4v"
            ),
            fps,
            (width, height)
        )

        success, frame = cap.read()

        if not success:
            raise RuntimeError(
                "Video contains no readable frames."
            )

        # First frame is used as reference
        writer.write(frame)

        previous_tensor = preprocess_frame(
            frame
        )

        frame_count = 1

        progress = st.progress(0)

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
                    outputs["reconstructed_frame"],
                    width,
                    height
                )

                writer.write(
                    reconstructed
                )

                previous_tensor = (
                    current_tensor
                )

                frame_count += 1

                progress.progress(
                    min(
                        frame_count / 1000,
                        1.0
                    )
                )

        cap.release()
        writer.release()

        st.success(
            f"Processing complete: "
            f"{frame_count} frames"
        )

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
                frame_count
            )

        st.video(
            output_path
        )

        with open(
            output_path,
            "rb"
        ) as file:

            st.download_button(
                "Download reconstructed video",
                data=file,
                file_name="reconstructed_video.mp4",
                mime="video/mp4"
            )

    except Exception as error:

        st.error(
            f"Processing failed: {error}"
        )
