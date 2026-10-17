
from pathlib import Path
import torch
import torch.nn.functional as F

from src.model_v2 import MiniGPTv2


ROOT = Path(__file__).resolve().parent
CHECKPOINT = ROOT / "minigpt_v2.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate(model, prompt, stoi, itos, max_new_tokens=150,
             temperature=0.6, top_k=5, seed=42):
    unknown = sorted(set(prompt) - set(stoi))
    if unknown:
        raise ValueError(
            f"Prompt contains characters outside the vocabulary: {unknown}"
        )

    generator = torch.Generator(device=DEVICE)
    generator.manual_seed(seed)

    ids = torch.tensor(
        [[stoi[ch] for ch in prompt]],
        dtype=torch.long,
        device=DEVICE,
    )

    model.eval()

    with torch.no_grad():
        for _ in range(max_new_tokens):
            context = ids[:, -model.max_length:]
            logits = model(context)[:, -1, :] / temperature

            k = min(top_k, logits.shape[-1])
            values, _ = torch.topk(logits, k=k, dim=-1)
            logits[logits < values[:, [-1]]] = float("-inf")

            probabilities = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(
                probabilities, num_samples=1, generator=generator
            )
            ids = torch.cat([ids, next_id], dim=1)

    return "".join(itos[int(i)] for i in ids[0].tolist())


def main():
    if not CHECKPOINT.exists():
        raise FileNotFoundError(
            "v2 checkpoint not found. Run python train_v2.py first."
        )

    checkpoint = torch.load(
        CHECKPOINT, map_location=DEVICE, weights_only=False
    )

    model = MiniGPTv2(
        vocab_size=checkpoint["vocab_size"],
        embed_size=checkpoint["embed_size"],
        num_layers=checkpoint["num_layers"],
        num_heads=checkpoint["num_heads"],
        dropout=checkpoint["dropout"],
        max_length=checkpoint["max_length"],
    ).to(DEVICE)

    model.load_state_dict(checkpoint["model_state_dict"])

    stoi = checkpoint["tokenizer_stoi"]
    itos = checkpoint["tokenizer_itos"]

    print("MiniGPT v2 loaded successfully.")
    print(f"Best epoch: {checkpoint['best_epoch']}")
    print(f"Validation loss: {checkpoint['val_loss']:.4f}")
    print(f"Validation perplexity: {checkpoint['val_perplexity']:.2f}")
    print(f"Parameters: {checkpoint['parameters']:,}")

    for prompt in [
        "Machine learning",
        "Artificial intelligence",
        "A neural network",
    ]:
        print(f"\nPrompt: {prompt}")
        try:
            print(
                generate(
                    model, prompt, stoi, itos,
                    max_new_tokens=150,
                    temperature=0.6,
                    top_k=5,
                    seed=42,
                )
            )
        except ValueError as error:
            print(f"Cannot test this prompt: {error}")


if __name__ == "__main__":
    main()
