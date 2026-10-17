import torch

from model import MiniGPT


# ============================================================
# 1. LOAD CHECKPOINT
# ============================================================

checkpoint = torch.load(
    "minigpt.pt",
    map_location="cpu"
)

vocab_size = checkpoint["vocab_size"]

stoi = checkpoint["tokenizer_stoi"]
itos = checkpoint["tokenizer_itos"]


# ============================================================
# 2. RECREATE MODEL
# ============================================================

model = MiniGPT(
    vocab_size=vocab_size,
    embed_size=checkpoint["embed_size"],
    num_layers=checkpoint["num_layers"],
    num_heads=checkpoint["num_heads"],
    forward_expansion=checkpoint["forward_expansion"],
    dropout=checkpoint["dropout"],
    max_length=checkpoint["max_length"]
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()


# ============================================================
# 3. TEXT GENERATION
# ============================================================

def generate(
    start_text,
    max_new_tokens=100,
    temperature=1.0,
    top_k=None
):

    # Convert prompt to token IDs
    tokens = []

    for ch in start_text:

        if ch in stoi:
            tokens.append(stoi[ch])
        else:
            print(
                f"Warning: '{ch}' is not in vocabulary."
            )

    if len(tokens) == 0:
        raise ValueError(
            "No characters from the prompt are in the vocabulary."
        )

    x = torch.tensor(
        [tokens],
        dtype=torch.long
    )

    for _ in range(max_new_tokens):

        # Use only the latest context
        x_input = x[
            :, -checkpoint["max_length"] :
        ]

        with torch.no_grad():

            logits = model(x_input)

        # Last token prediction
        logits = logits[:, -1, :]

        # ----------------------------------------------------
        # Temperature
        # ----------------------------------------------------

        logits = logits / temperature

        # ----------------------------------------------------
        # Top-k filtering
        # ----------------------------------------------------

        if top_k is not None:

            values, indices = torch.topk(
                logits,
                min(top_k, logits.size(-1))
            )

            filtered_logits = torch.full_like(
                logits,
                float("-inf")
            )

            filtered_logits.scatter_(
                1,
                indices,
                values
            )

            logits = filtered_logits

        # Convert to probabilities
        probabilities = torch.softmax(
            logits,
            dim=-1
        )

        # Sample next token
        next_token = torch.multinomial(
            probabilities,
            num_samples=1
        )

        # Add token to sequence
        x = torch.cat(
            [x, next_token],
            dim=1
        )

    # Decode
    generated_text = ""

    for token in x[0].tolist():

        generated_text += itos[token]

    return generated_text


# ============================================================
# 4. TEST DIFFERENT SETTINGS
# ============================================================

if __name__ == "__main__":

    prompt = "Machine"

    print("\n" + "=" * 60)
    print("TEMPERATURE = 0.5")
    print("=" * 60)

    print(
        generate(
            prompt,
            max_new_tokens=100,
            temperature=0.5,
            top_k=5
        )
    )

    print("\n" + "=" * 60)
    print("TEMPERATURE = 1.0")
    print("=" * 60)

    print(
        generate(
            prompt,
            max_new_tokens=100,
            temperature=1.0,
            top_k=5
        )
    )

    print("\n" + "=" * 60)
    print("TEMPERATURE = 1.5")
    print("=" * 60)

    print(
        generate(
            prompt,
            max_new_tokens=100,
            temperature=1.5,
            top_k=5
        )
    )