import torch
import torch.nn as nn
from multi_head_attention import MultiHeadAttention


class TransformerBlock(nn.Module):

    def __init__(self, embed_size, num_heads, dropout, forward_expansion):
        super().__init__()

        # Multi-Head Attention
        self.attention = MultiHeadAttention(
            embed_size,
            num_heads
        )

        # Layer Normalization
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        # Feed Forward Network
        self.feed_forward = nn.Sequential(
            nn.Linear(
                embed_size,
                forward_expansion * embed_size
            ),

            nn.ReLU(),

            nn.Linear(
                forward_expansion * embed_size,
                embed_size
            )
        )

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query):

        # Multi-Head Attention
        attention = self.attention(query)

        # Residual connection + normalization
        x = self.dropout(attention) + query
        x = self.norm1(x)

        # Feed Forward Network
        forward = self.feed_forward(x)

        # Residual connection + normalization
        x = self.dropout(forward) + x
        x = self.norm2(x)

        return x


# Test
if __name__ == "__main__":

    batch_size = 2
    sequence_length = 4
    embed_size = 8

    num_heads = 2
    dropout = 0.1
    forward_expansion = 4

    x = torch.randn(
        batch_size,
        sequence_length,
        embed_size
    )

    model = TransformerBlock(
        embed_size,
        num_heads,
        dropout,
        forward_expansion
    )

    output = model(x, x, x)

    print("Input shape :", x.shape)
    print("Output shape:", output.shape)