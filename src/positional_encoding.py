import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):

    def __init__(self, embed_size, max_length=100):

        super().__init__()

        # Create a matrix for positional encoding
        pe = torch.zeros(max_length, embed_size)

        # Position of each token
        position = torch.arange(
            0,
            max_length,
            dtype=torch.float
        ).unsqueeze(1)

        # Different frequencies for different dimensions
        div_term = torch.exp(
            torch.arange(
                0,
                embed_size,
                2
            ).float()
            * (-math.log(10000.0) / embed_size)
        )

        # Apply sine to even dimensions
        pe[:, 0::2] = torch.sin(
            position * div_term
        )

        # Apply cosine to odd dimensions
        pe[:, 1::2] = torch.cos(
            position * div_term
        )

        # Add batch dimension
        pe = pe.unsqueeze(0)

        # Store as a non-trainable buffer
        self.register_buffer("pe", pe)

    def forward(self, x):

        sequence_length = x.size(1)

        # Add positional information
        x = x + self.pe[:, :sequence_length]

        return x


# Test
if __name__ == "__main__":

    batch_size = 2
    sequence_length = 4
    embed_size = 8

    x = torch.zeros(
        batch_size,
        sequence_length,
        embed_size
    )

    positional_encoding = PositionalEncoding(
        embed_size
    )

    output = positional_encoding(x)

    print("Input shape :", x.shape)
    print("Output shape:", output.shape)

    print("\nPositional Encoding:")
    print(output[0])