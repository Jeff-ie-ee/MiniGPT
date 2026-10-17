import torch
import torch.nn as nn
import math


class MultiHeadAttention(nn.Module):

    def __init__(self, embed_size, num_heads):
        super().__init__()

        assert embed_size % num_heads == 0

        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads

        # Learnable projections
        self.query = nn.Linear(embed_size, embed_size)
        self.key = nn.Linear(embed_size, embed_size)
        self.value = nn.Linear(embed_size, embed_size)

        # Final projection
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, x):

        batch_size, sequence_length, embed_size = x.shape

        # Create Q, K, V
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # Split into attention heads
        Q = Q.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim
        )

        K = K.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim
        )

        V = V.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim
        )

        # Move heads before sequence dimensions
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # Scaled dot-product attention
        scores = Q @ K.transpose(-2, -1)

        scores = scores / math.sqrt(self.head_dim)

        # ------------------------------------------------
        # Causal mask
        # ------------------------------------------------
        mask = torch.tril(
            torch.ones(
                sequence_length,
                sequence_length,
                device=x.device
            )
        )

        # Hide future tokens
        scores = scores.masked_fill(
            mask == 0,
            float("-inf")
        )

        # Convert to probabilities
        attention = torch.softmax(
            scores,
            dim=-1
        )

        # Weighted sum of values
        out = attention @ V

        # Combine heads
        out = out.transpose(1, 2)

        out = out.contiguous().view(
            batch_size,
            sequence_length,
            self.embed_size
        )

        # Final projection
        out = self.fc_out(out)

        return out


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    batch_size = 2
    sequence_length = 4
    embed_size = 8
    num_heads = 2

    x = torch.randn(
        batch_size,
        sequence_length,
        embed_size
    )

    model = MultiHeadAttention(
        embed_size,
        num_heads
    )

    output = model(x)

    print("Input shape :", x.shape)
    print("Output shape:", output.shape)