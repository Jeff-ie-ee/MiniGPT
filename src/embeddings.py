import torch
import torch.nn as nn


vocab_size = 50
embed_size = 8

embedding = nn.Embedding(
    vocab_size,
    embed_size
)

tokens = torch.tensor([
    [1, 5, 7, 3]
])

output = embedding(tokens)

print("Token shape :", tokens.shape)
print("Embedding shape:", output.shape)
print("\nEmbeddings:")
print(output)