
import torch
import torch.nn as nn
import torch.nn.functional as F


def warp_frame(frame, flow):
    # Differentiable frame warping

    B, C, H, W = frame.shape

    y, x = torch.meshgrid(
        torch.arange(H, device=frame.device),
        torch.arange(W, device=frame.device),
        indexing="ij"
    )

    grid = torch.stack(
        (x, y),
        dim=0
    ).float()

    grid = grid.unsqueeze(0).expand(
        B, -1, -1, -1
    )

    sampling_grid = grid + flow

    sampling_grid[:, 0] = (
        2.0 * sampling_grid[:, 0]
        / max(W - 1, 1)
        - 1.0
    )

    sampling_grid[:, 1] = (
        2.0 * sampling_grid[:, 1]
        / max(H - 1, 1)
        - 1.0
    )

    sampling_grid = sampling_grid.permute(
        0, 2, 3, 1
    )

    warped_frame = F.grid_sample(
        frame,
        sampling_grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True
    )

    return warped_frame


class MotionEstimationNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Conv2d(
                6, 32,
                kernel_size=7,
                stride=2,
                padding=3
            ),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                32, 64,
                kernel_size=5,
                stride=2,
                padding=2
            ),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                64, 96,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                96, 64,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(
                64, 32,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(
                32, 16,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                16, 2,
                kernel_size=3,
                stride=1,
                padding=1
            )
        )

    def forward(
        self,
        previous_frame,
        current_frame
    ):
        x = torch.cat(
            [
                previous_frame,
                current_frame
            ],
            dim=1
        )

        return self.network(x)


class ResidualEncoder(nn.Module):
    def __init__(self, latent_channels=64):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(
                3, 64,
                kernel_size=5,
                stride=2,
                padding=2
            ),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                64, 128,
                kernel_size=5,
                stride=2,
                padding=2
            ),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                128,
                latent_channels,
                kernel_size=5,
                stride=2,
                padding=2
            )
        )

    def forward(self, residual):
        return self.encoder(residual)


class ResidualDecoder(nn.Module):
    def __init__(self, latent_channels=64):
        super().__init__()

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                latent_channels,
                128,
                kernel_size=5,
                stride=2,
                padding=2,
                output_padding=1
            ),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(
                128, 64,
                kernel_size=5,
                stride=2,
                padding=2,
                output_padding=1
            ),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(
                64, 3,
                kernel_size=5,
                stride=2,
                padding=2,
                output_padding=1
            )
        )

    def forward(self, latent):
        return self.decoder(latent)


class EntropyModel(nn.Module):
    def __init__(self, latent_channels=64):
        super().__init__()

        self.log_scale = nn.Parameter(
            torch.zeros(latent_channels)
        )

    def forward(self, latent):
        scale = torch.exp(
            self.log_scale
        ).view(1, -1, 1, 1)

        return torch.clamp(
            scale,
            min=1e-3,
            max=10.0
        )


class NeuralVideoCompressionModel(nn.Module):
    def __init__(self, latent_channels=64):
        super().__init__()

        self.motion_net = (
            MotionEstimationNetwork()
        )

        self.residual_encoder = (
            ResidualEncoder(
                latent_channels
            )
        )

        self.residual_decoder = (
            ResidualDecoder(
                latent_channels
            )
        )

        self.entropy_model = (
            EntropyModel(
                latent_channels
            )
        )

    def forward(
        self,
        previous_frame,
        current_frame
    ):
        # Motion estimation
        flow = self.motion_net(
            previous_frame,
            current_frame
        )

        # Motion compensation
        warped_frame = warp_frame(
            previous_frame,
            flow
        )

        # Residual
        residual = (
            current_frame - warped_frame
        )

        # Residual encoding
        latent = self.residual_encoder(
            residual
        )

        # Straight-through quantization
        quantized_latent = (
            latent
            + (
                torch.round(latent)
                - latent
            ).detach()
        )

        # Residual decoding
        reconstructed_residual = (
            self.residual_decoder(
                quantized_latent
            )
        )

        # Final reconstruction
        reconstructed_frame = torch.clamp(
            warped_frame
            + reconstructed_residual,
            0.0,
            1.0
        )

        return {
            "flow": flow,
            "warped_frame": warped_frame,
            "residual": residual,
            "latent": latent,
            "quantized_latent": quantized_latent,
            "reconstructed_residual":
                reconstructed_residual,
            "reconstructed_frame":
                reconstructed_frame
        }
