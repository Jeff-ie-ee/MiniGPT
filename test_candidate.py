
import torch
from pathlib import Path

from src.model import MiniGPT

BASE_DIR = Path(__file__).resolve().parent
CHECKPOINT_PATH = BASE_DIR / "minigpt_candidate.pt"

torch.manual_seed(42)

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location="cpu",
    weights_only=False
)

model = MiniGPT(
    vocab_size=checkpoint["vocab_size"],
    embed_size=checkpoint["embed_size"],
    num_layers=checkpoint["num_layers"],
    num_heads=checkpoint["num_heads"],
    forward_expansion=checkpoint["forward_expansion"],
    dropout=checkpoint["dropout"],
    max_length=checkpoint["max_length"]
)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

stoi = checkpoint["tokenizer_stoi"]
itos = checkpoint["tokenizer_itos"]
context_length = checkpoint["max_length"]


@torch.inference_mode()
def generate(prompt, max_new_tokens=100, temperature=0.6, top_k=5):
    unknown = sorted(set(prompt) - set(stoi))
    if unknown:
        raise ValueError(f"Characters missing from vocabulary: {unknown}")

    tokens = [stoi[ch] for ch in prompt]
    x = torch.tensor([tokens], dtype=torch.long)
    generator = torch.Generator(device="cpu").manual_seed(42)

    for _ in range(max_new_tokens):
        x_input = x[:, -context_length:]
        logits = model(x_input)
        logits = logits[:, -1, :] / temperature

        k = min(top_k, logits.size(-1))
        values, _ = torch.topk(logits, k)
        logits[logits < values[:, [-1]]] = float("-inf")

        probabilities = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(
            probabilities, num_samples=1, generator=generator
        )

        x = torch.cat([x, next_token], dim=1)

    return "".join(itos[int(token)] for token in x[0])


if __name__ == "__main__":
    print("Candidate model loaded successfully.")
    print("Best epoch:", checkpoint["best_epoch"])
    print("Validation loss:", round(checkpoint["val_loss"], 4))
    print("\nGenerated sample:\n")
    print(generate("Machine learning"))
