# Neural Video Compression

A deep learning based video compression prototype built entirely
from scratch using PyTorch.

## Overview

This project explores learned video compression using temporal
redundancy between consecutive video frames.

The system consists of:

- CNN-based motion estimation
- Differentiable motion compensation
- Residual computation
- Learned residual encoder and decoder
- Latent quantization
- Gaussian entropy model
- Rate-distortion optimization
- Video reconstruction and evaluation

## Architecture

Previous Frame + Current Frame
        |
        v
Motion Estimation Network
        |
        v
Optical Flow
        |
        v
Motion Compensation
        |
        v
Residual
        |
        v
Residual Encoder
        |
        v
Latent Representation
        |
        v
Quantization
        |
        v
Residual Decoder
        |
        v
Reconstructed Frame

## Training

The neural networks are trained from scratch using randomly
initialized parameters.

No pretrained models or pretrained weights are used.

The training objective is:

L = MSE + lambda x BPP

where MSE measures reconstruction distortion and BPP represents
the learned bitrate estimate.

## Dataset

Training and evaluation use a subset of the Vimeo-90K video
dataset containing consecutive video frames.

The dataset itself is not included in this repository.

## Results

Current test-set results:

- PSNR: 27.92 dB
- MSE: 0.001615
- Learned BPP: 1.171961

## Demo

A Streamlit application is included for testing the trained model
on uploaded videos.

## Project Structure

neural-video-compression/
├── app/
│   ├── app.py
│   └── neural_codec.py
├── checkpoints/
│   └── best_model.pth
├── data/
├── evaluation/
├── inference/
├── models/
├── results/
├── training/
├── utils/
├── notebooks/
├── docs/
├── requirements.txt
├── .gitignore
└── README.md

## Limitations

The current implementation uses a learned bitrate estimate rather
than a fully arithmetic-coded neural bitstream.

Video frames are processed internally at the model's training
resolution.

Future improvements can include more advanced entropy models,
better motion estimation, perceptual losses, and actual bitstream
generation.

## License

This project is intended for academic and educational use.
