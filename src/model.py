import torch
import torch.nn as nn

try:
    from src.multi_head_attention import MultiHeadAttention
    from src.positional_encoding import PositionalEncoding
except ModuleNotFoundError:
    from multi_head_attention import MultiHeadAttention
    from positional_encoding import PositionalEncoding


class TransformerBlock(nn.Module):

    def __init__(
        self,
        embed_size,
        num_heads,
        dropout,
        forward_expansion
    ):
        super().__init__()

        self.attention = MultiHeadAttention(
            embed_size,
            num_heads
        )

        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

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

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        attention = self.attention(x)

        x = self.norm1(
            x + self.dropout(attention)
        )

        forward = self.feed_forward(x)

        x = self.norm2(
            x + self.dropout(forward)
        )

        return x


class MiniGPT(nn.Module):

    def __init__(
        self,
        vocab_size,
        embed_size=128,
        num_layers=4,
        num_heads=4,
        forward_expansion=4,
        dropout=0.1,
        max_length=256
    ):
        super().__init__()

        self.token_embedding = nn.Embedding(
            vocab_size,
            embed_size
        )

        self.position_encoding = PositionalEncoding(
            embed_size,
            max_length
        )

        self.dropout = nn.Dropout(dropout)

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                embed_size,
                num_heads,
                dropout,
                forward_expansion
            )
            for _ in range(num_layers)
        ])

        self.lm_head = nn.Linear(
            embed_size,
            vocab_size
        )

    def forward(self, x):

        x = self.token_embedding(x)

        x = self.position_encoding(x)

        x = self.dropout(x)

        for block in self.transformer_blocks:
            x = block(x)

        logits = self.lm_head(x)

        return logits


if __name__ == "__main__":

    vocab_size = 44

    model = MiniGPT(
        vocab_size=vocab_size,
        embed_size=64,
        num_layers=1,
        num_heads=2,
        forward_expansion=4,
        dropout=0.1,
        max_length=32
    )

    x = torch.randint(
        0,
        vocab_size,
        (2, 32)
    )

    output = model(x)

    print("Input shape :", x.shape)
    print("Output shape:", output.shape)