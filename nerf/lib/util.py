import torch
import torch.nn as nn
import math
import numpy as np

def get_subdict(named_params, key):

    if named_params is None: return None

    subdict = {}

    for name, param in named_params.items():
        if name.startswith(key):
            new_name = name[len(key):]
            subdict[new_name] = param

    return subdict


def get_coordinate_grid(batch_size, dimensions, device="cpu"):
    """
    :param dimensions: Height, Width, [Depth]
    return: B, C, Height, Width, [Depth]
    """

    if len(dimensions) == 2:
        x = torch.linspace(-1, 1, dimensions[1])
        y = torch.linspace(-1, 1, dimensions[0])
        X, Y = torch.meshgrid(x, y, indexing='xy')
        coords = torch.stack([X, Y], dim=0)
    elif len(dimensions) == 3:
        raise NotImplementedError
    else:
        raise ValueError('Dimensions not supported')

    coords = coords.unsqueeze(0).to(device).repeat_interleave(batch_size, dim=0)

    return coords


class PositionalEncoding(nn.Module):
    def __init__(self, input_channels: int, output_features: int):
        """
        Positional Encoding Module for Implicit Neural Representations.

        Args:
            input_channels (int): Number of input channels (dimensionality of input).
            output_features (int): Number of output features after positional encoding.
        """
        super(PositionalEncoding, self).__init__()
        self.input_channels = input_channels
        self.output_features = output_features
        self.num_frequencies = output_features // (2 * input_channels)

        # Ensure output_features is a multiple of 2 * input_channels
        if self.output_features % (2 * input_channels) != 0:
            raise ValueError(
                "output_features must be a multiple of 2 * input_channels"
            )

        # Create frequency bands for encoding
        self.freq_bands = torch.linspace(1.0, 2 ** (self.num_frequencies - 1), self.num_frequencies)

    def forward(self, x):
        """
        Forward pass for positional encoding.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, n_channels, n_pixels).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_features, n_pixels).
        """
        batch_size, n_channels, n_pixels = x.shape

        if n_channels != self.input_channels:
            raise ValueError(
                f"Expected input with {self.input_channels} channels, but got {n_channels} channels."
            )

        # Reshape for broadcasting with frequency bands
        x = x.unsqueeze(3)  # Shape: (batch_size, n_channels, n_pixels, 1)
        freq_bands = self.freq_bands.to(x.device).view(1, 1, 1, -1)  # Shape: (1, 1, 1, num_frequencies)

        # Compute sinusoidal positional encodings
        sin = torch.sin(2 * math.pi * x * freq_bands)  # Shape: (batch_size, n_channels, n_pixels, num_frequencies)
        cos = torch.cos(2 * math.pi * x * freq_bands)  # Shape: (batch_size, n_channels, n_pixels, num_frequencies)

        # Concatenate sin and cos along the frequency dimension
        encoded = torch.cat([sin, cos], dim=3)  # Shape: (batch_size, n_channels, n_pixels, 2 * num_frequencies)

        # Flatten the channel and frequency dimensions
        encoded = encoded.view(batch_size, n_channels * 2 * self.num_frequencies,
                               n_pixels)  # Shape: (batch_size, output_features, n_pixels)

        return encoded