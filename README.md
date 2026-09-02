# Neural Video Compression

A PyTorch-based neural video compression prototype built entirely from scratch using learned motion estimation, residual coding, entropy modeling, and rate-distortion optimization.

---

## Overview

Video contains substantial **temporal redundancy**: consecutive frames often share most of their visual information.

Instead of independently encoding every frame, this project learns to predict the current frame from the previous frame and encode only the information that cannot be predicted.

The system consists of:

- CNN-based motion estimation
- Differentiable motion compensation
- Residual computation
- Learned residual encoder and decoder
- Latent quantization
- Learned Gaussian entropy model
- Rate-distortion optimization
- Video reconstruction
- PSNR, MSE, and BPP evaluation

> **All neural networks in this project are trained from scratch with randomly initialized parameters. No pretrained models or pretrained weights are used.**

---

## System Architecture

```mermaid
flowchart LR

    A["Previous Frame"]
    B["Current Frame"]

    A --> C["Motion Estimation CNN"]
    B --> C

    C --> D["Optical Flow"]

    A --> E["Differentiable Warping"]
    D --> E

    E --> F["Motion Prediction"]

    B --> G["Residual"]
    F --> G

    G --> H["Residual Encoder"]
    H --> I["Latent Representation"]
    I --> J["Quantization"]

    J --> K["Residual Decoder"]
    K --> L["Reconstructed Residual"]

    F --> M["Reconstructed Frame"]
    L --> M

How the Model Works
1. Motion Estimation

Two consecutive frames are given to a convolutional neural network:

$$ (F_{t-1}, F_t) \rightarrow \text{Motion Network} \rightarrow \mathbf{f}_t $$

where $\mathbf{f}_t$ is a two-channel optical-flow field containing horizontal and vertical motion information.

The motion network is trained entirely from scratch.

2. Motion Compensation

The predicted flow is used to warp the previous frame:

$$ \hat{F}_t^{motion} = Warp(F_{t-1}, \mathbf{f}_t) $$

This creates a prediction of the current frame using information from the previous frame.

3. Residual Computation

The part that cannot be explained by motion is represented as a residual:

$$ R_t = F_t - \hat{F}_t^{motion} $$

A better motion prediction should produce a smaller and easier-to-compress residual.

4. Residual Encoding

The residual is passed through a learned encoder:

$$ R_t \xrightarrow{Encoder} y_t $$

where $y_t$ is the latent representation.

The encoder reduces the spatial resolution while learning a compact representation of the residual information.

5. Quantization

The continuous latent representation is quantized:

$$ \hat{y}_t = round(y_t) $$

A straight-through estimator is used during training so that the network can still receive gradients through the quantization operation.

6. Residual Decoding

The quantized latent is passed through the learned decoder:

$$ \hat{y}_t \xrightarrow{Decoder} \hat{R}_t $$

where $\hat{R}_t$ is the reconstructed residual.

7. Frame Reconstruction

The final frame is reconstructed by adding the predicted frame and decoded residual:

$$ \hat{F}_t = \hat{F}_t^{motion} + \hat{R}_t $$

The reconstructed result is clipped to the valid image range.

Entropy Modeling

The latent representation should not only reconstruct the image accurately; it should also be efficient to encode.

We model the latent values using a learned Gaussian distribution.

For a quantized latent value $\hat{y}$:

$$ p(\hat{y}) = \Phi \left( \frac{\hat{y}+0.5}{\sigma} \right) - \Phi \left( \frac{\hat{y}-0.5}{\sigma} \right) $$

where:

$\Phi$ is the standard Gaussian cumulative distribution function.
$\sigma$ is a learned scale parameter.

The estimated number of bits is:

$$ R = -\log_2 p(\hat{y}) $$

The total rate is then converted into Bits Per Pixel (BPP):

$$ BPP = \frac{\text{Estimated Total Bits}} {\text{Number of Image Pixels}} $$

The current implementation uses a learned bitrate estimate. It does not yet generate a standards-compatible arithmetic-coded neural bitstream.

Rate-Distortion Optimization

The model jointly optimizes reconstruction quality and estimated bitrate.

The training objective is:

$$ \mathcal{L} = D + \lambda R $$

In this implementation:

$$ \mathcal{L} = MSE + \lambda \cdot BPP $$

where:

$MSE$ represents reconstruction distortion.
$BPP$ represents the estimated bitrate.
$\lambda$ controls the trade-off between visual quality and compression rate.

A larger $\lambda$ places more emphasis on reducing bitrate, while a smaller $\lambda$ places more emphasis on reconstruction quality.

Training Setup

The model was trained using:

Component	Configuration
Framework	PyTorch
GPU	NVIDIA Tesla T4
Dataset	Vimeo-90K-derived mini subset
Training sequences	800
Validation sequences	100
Test sequences	100
Input frame size	448 × 256
Batch size	4
Optimizer	Adam
Learning rate	$10^{-4}$
Weight decay	$10^{-5}$
Training epochs	10

The dataset split is performed at the sequence level to avoid placing frames from the same sequence in both training and evaluation sets.

Dataset

A subset containing 1,000 video sequences was used:

800 training sequences
100 validation sequences
100 test sequences

Each sequence contains consecutive RGB frames.

The dataset is not included in this repository because of its size.

Evaluation Metrics
MSE

Mean Squared Error measures the average pixel-level reconstruction error:

$$ MSE = \frac{1}{N} \sum_{i=1}^{N} (x_i-\hat{x}_i)^2 $$

Lower MSE indicates lower reconstruction error.

PSNR

Peak Signal-to-Noise Ratio is derived from MSE:

$$ PSNR = 10\log_{10} \left( \frac{MAX^2}{MSE} \right) $$

For normalized images, $MAX=1$.

Higher PSNR generally indicates better reconstruction quality.

BPP

Bits Per Pixel represents the estimated number of coding bits required per image pixel:

$$ BPP = \frac{\text{Estimated Bits}} {\text{Number of Pixels}} $$

Lower BPP indicates a lower estimated bitrate.

Experimental Results
Held-Out Test Set
Metric	Result
PSNR	27.92 dB
MSE	0.001615
Learned BPP	1.171961

These values were obtained using the best checkpoint selected using validation loss.

Reconstruction Example

The comparison shows the original target frame and the reconstruction generated by the trained neural codec.

Real Video Inference

The trained model was also tested on a separate video containing approximately 1,691 readable frames.

The inference pipeline performs:

Input Video
     ↓
Frame Extraction
     ↓
Motion Estimation
     ↓
Motion Compensation
     ↓
Residual Computation
     ↓
Residual Encoding
     ↓
Latent Quantization
     ↓
Residual Decoding
     ↓
Frame Reconstruction
     ↓
Output Video

The model successfully processed the full test video and generated a reconstructed MP4.

Streamlit Demo

The repository contains a Streamlit application for testing the trained neural codec on uploaded videos.

The application:

Accepts a video upload.
Loads the trained checkpoint.
Processes the video frame by frame.
Reconstructs each frame using the neural codec.
Produces a reconstructed MP4.

The application uses the trained model stored in:

checkpoints/best_model.pth
Project Structure
Neural-Video-Compression/
│
├── app/
│   ├── app.py
│   └── neural_codec.py
│
├── checkpoints/
│   └── best_model.pth
│
├── notebooks/
│   └── Neural_Video_Compression.ipynb
│
├── results/
│   ├── final_test_results.json
│   ├── real_video_evaluation.json
│   ├── test_reconstruction_comparison.png
│   └── training_history.json
│
├── .gitignore
├── README.md
└── requirements.txt
Reproducibility

The complete development process is documented in:

notebooks/Neural_Video_Compression.ipynb

The trained checkpoint is included in:

checkpoints/best_model.pth

Dependencies are listed in:

requirements.txt

The dataset itself must be obtained separately.

Limitations

This project is an academic/research prototype rather than a production video codec.

Current limitations include:

The entropy model provides a learned bitrate estimate rather than a complete arithmetic-coded bitstream.
The neural model operates internally at a fixed training resolution of 448 × 256.
Reconstructed frames can show some smoothing and visual artifacts.
The current implementation is designed for demonstrating the principles of learned video compression rather than real-time deployment.
Future Work

Possible extensions include:

More advanced motion estimation networks
Improved entropy models
Perceptual loss functions such as SSIM or LPIPS
Multi-scale video processing
Actual entropy coding and neural bitstream generation
Rate-control experiments with different $\lambda$ values
Faster inference and GPU deployment
Improved video/audio preservation
Technologies
Python
PyTorch
OpenCV
NumPy
Pillow
FFmpeg
Streamlit
Google Colab
Author

Munna Kumar Sah

GitHub: @doitmuna

License

This project is intended for academic and educational use.


### Why this version is better

The equations such as

\[
R_t = F_t-\hat F_t^{motion}
\]

and

\[
\mathcal{L}=MSE+\lambda\cdot BPP
\]

will render as proper mathematical expressions on GitHub rather than appearing as raw text. GitHub officially supports block math with `$$...$$`, and Mermaid diagrams inside ` ```mermaid ` blocks. :contentReference[oaicite:1]{index=1}

### One important correction

I changed the wording from **“compressed video”** to **“reconstructed video”** in places where appropriate. That's deliberate: our current implementation has a learned BPP estimate but does **not yet produce a true neural compressed bitstream**. Being precise about this will make the project more defensible when an RA/TA reviewer examines the code.

After pasting this into `README.md`, save it and run:

```powershell
# Check the README change
git status