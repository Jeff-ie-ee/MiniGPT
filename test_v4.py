
from pathlib import Path
import torch

from src.model_v2 import MiniGPTv2

ROOT = Path(__file__).resolve().parent
CHECKPOINT = ROOT / "minigpt_v4.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    if not CHECKPOINT.exists():
        raise FileNotFoundError(
            "Checkpoint not found. Run python train_v4.py first."
        )

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=DEVICE,
        weights_only=True,
    )

    config = checkpoint["config"]

    model = MiniGPTv2(
        vocab_size=config["vocab_size"],
        embed_size=config["embed_size"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        dropout=config["dropout"],
        max_length=config["max_length"],
    ).to(DEVICE)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    stoi = checkpoint["stoi"]
    itos = {int(key): value for key, value in checkpoint["itos"].items()}

    print("MiniGPT v4 loaded successfully.")
    print(f"Best epoch: {checkpoint['best_epoch']}")
    print(f"Validation loss: {checkpoint['best_val_loss']:.4f}")

    prompts = [
        "Machine learning",
        "Artificial intelligence",
        "A neural network",
    ]

    for prompt in prompts:
        token_ids = [stoi[ch] for ch in prompt if ch in stoi]

        if not token_ids:
            print(f"\nNo recognized characters in prompt: {prompt}")
            continue

        x = torch.tensor(
            [token_ids],
            dtype=torch.long,
            device=DEVICE,
        )

        generator = torch.Generator(device=DEVICE)
        generator.manual_seed(42)

        with torch.no_grad():
            for _ in range(120):
                context = x[:, -config["max_length"]:]
                logits = model(context)
                next_logits = logits[:, -1, :] / 0.7

                k = min(5, next_logits.size(-1))
                top_values, _ = torch.topk(next_logits, k)
                next_logits[
                    next_logits < top_values[:, [-1]]
                ] = float("-inf")

                probabilities = torch.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(
                    probabilities,
                    num_samples=1,
                    generator=generator,
                )
                x = torch.cat([x, next_token], dim=1)

        generated = "".join(
            itos[int(token)] for token in x[0].tolist()
        )

        print(f"\nPrompt: {prompt}")
        print(generated)
        print("-" * 60)


if __name__ == "__main__":
    main()
