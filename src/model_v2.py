
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, embed_size, num_heads, dropout):
        super().__init__()

        if embed_size % num_heads != 0:
            raise ValueError("embed_size must be divisible by num_heads")

        self.num_heads = num_heads
        self.head_size = embed_size // num_heads

        self.qkv = nn.Linear(embed_size, 3 * embed_size)
        self.projection = nn.Linear(embed_size, embed_size)

        self.attention_dropout = nn.Dropout(dropout)
        self.residual_dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size, sequence_length, embed_size = x.shape

        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(
            batch_size, sequence_length,
            self.num_heads, self.head_size
        ).transpose(1, 2)

        k = k.view(
            batch_size, sequence_length,
            self.num_heads, self.head_size
        ).transpose(1, 2)

        v = v.view(
            batch_size, sequence_length,
            self.num_heads, self.head_size
        ).transpose(1, 2)

        # Causal attention prevents looking at future characters.
        output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=(
                self.attention_dropout.p if self.training else 0.0
            ),
            is_causal=True,
        )

        output = output.transpose(1, 2).contiguous()
        output = output.view(
            batch_size, sequence_length, embed_size
        )

        return self.residual_dropout(self.projection(output))


class TransformerBlock(nn.Module):
    def __init__(self, embed_size, num_heads, dropout):
        super().__init__()

        self.norm1 = nn.LayerNorm(embed_size)
        self.attention = CausalSelfAttention(
            embed_size, num_heads, dropout
        )

        self.norm2 = nn.LayerNorm(embed_size)
        self.feedforward = nn.Sequential(
            nn.Linear(embed_size, 4 * embed_size),
            nn.GELU(),
            nn.Linear(4 * embed_size, embed_size),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.feedforward(self.norm2(x))
        return x


class MiniGPTv2(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_size=64,
        num_layers=2,
        num_heads=4,
        dropout=0.1,
        max_length=64,
    ):
        super().__init__()

        self.max_length = max_length

        self.token_embedding = nn.Embedding(vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)
        self.embedding_dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_size, num_heads, dropout)
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(embed_size)
        self.lm_head = nn.Linear(embed_size, vocab_size, bias=False)

        # Tie token embeddings and output projection weights.
        self.lm_head.weight = self.token_embedding.weight

        self.apply(self._initialize_weights)

    @staticmethod
    def _initialize_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, token_ids):
        batch_size, sequence_length = token_ids.shape

        if sequence_length > self.max_length:
            raise ValueError(
                f"Input length {sequence_length} exceeds "
                f"context length {self.max_length}"
            )

        positions = torch.arange(
            sequence_length, device=token_ids.device
        )

        x = (
            self.token_embedding(token_ids)
            + self.position_embedding(positions)[None, :, :]
        )
        x = self.embedding_dropout(x)

        for block in self.blocks:
            x = block(x)

        x = self.final_norm(x)
        return self.lm_head(x)
