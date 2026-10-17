import torch
import torch.nn.functional as F


def self_attention(Q, K, V):
    """
    Calculate scaled dot-product self-attention.

    Q = Query
    K = Key
    V = Value
    """

    # 1. Calculate similarity between Query and Key
    scores = Q @ K.T

    # 2. Scale the scores
    d_k = K.size(-1)
    scores = scores / (d_k ** 0.5)

    # 3. Convert scores into attention weights
    attention_weights = F.softmax(scores, dim=-1)

    # 4. Weighted sum of Values
    output = attention_weights @ V

    return output, attention_weights


# Example
Q = torch.tensor([
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 1.0]
])

K = torch.tensor([
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 1.0]
])

V = torch.tensor([
    [10.0, 0.0],
    [0.0, 10.0],
    [5.0, 5.0]
])


output, weights = self_attention(Q, K, V)

print("Attention Weights:")
print(weights)

print("\nAttention Output:")
print(output)