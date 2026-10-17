
from pathlib import Path
import json
import math
import random
import time

import torch
import torch.nn as nn

from src.model_v2 import MiniGPTv2

ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "data" / "text_v4.txt"
CHECKPOINT_FILE = ROOT / "minigpt_v4.pt"
HISTORY_FILE = ROOT / "results" / "training_history_v4.json"

SEED = 42
BLOCK_SIZE = 64
BATCH_SIZE = 16
EPOCHS = 20
STEPS_PER_EPOCH = 100
EVAL_BATCHES = 30
LEARNING_RATE = 0.0005
WEIGHT_DECAY = 0.01
PATIENCE = 5

EMBED_SIZE = 64
NUM_LAYERS = 2
NUM_HEADS = 4
DROPOUT = 0.1

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    random.seed(SEED)
    torch.manual_seed(SEED)

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}\n"
            "First run: python build_dataset_v4.py"
        )

    text = DATA_FILE.read_text(encoding="utf-8")
    if len(text) < 1000:
        raise ValueError("Dataset is too small for this experiment.")

    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}

    encoded = torch.tensor(
        [stoi[ch] for ch in text],
        dtype=torch.long,
    )

    split = int(0.90 * len(encoded))
    train_data = encoded[:split]
    val_data = encoded[split:]

    if len(train_data) <= BLOCK_SIZE or len(val_data) <= BLOCK_SIZE:
        raise ValueError("Not enough data for the selected block size.")

    model = MiniGPTv2(
        vocab_size=len(chars),
        embed_size=EMBED_SIZE,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        dropout=DROPOUT,
        max_length=BLOCK_SIZE,
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    criterion = nn.CrossEntropyLoss()

    def get_batch(source):
        starts = torch.randint(
            0,
            len(source) - BLOCK_SIZE - 1,
            (BATCH_SIZE,),
        )

        x = torch.stack([
            source[i:i + BLOCK_SIZE] for i in starts
        ])
        y = torch.stack([
            source[i + 1:i + BLOCK_SIZE + 1] for i in starts
        ])

        return x.to(DEVICE), y.to(DEVICE)

    @torch.no_grad()
    def evaluate(source):
        model.eval()
        losses = []

        for _ in range(EVAL_BATCHES):
            x, y = get_batch(source)
            logits = model(x)

            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                y.reshape(-1),
            )
            losses.append(loss.item())

        model.train()
        return sum(losses) / len(losses)

    parameter_count = sum(
        parameter.numel() for parameter in model.parameters()
    )

    print(f"Device: {DEVICE}")
    print(f"Dataset: {DATA_FILE}")
    print(f"Characters: {len(text):,}")
    print(f"Vocabulary size: {len(chars)}")
    print(f"Training characters: {len(train_data):,}")
    print(f"Validation characters: {len(val_data):,}")
    print(f"Parameters: {parameter_count:,}")

    history = []
    best_val_loss = float("inf")
    best_epoch = 0
    patience_count = 0
    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_losses = []

        for _ in range(STEPS_PER_EPOCH):
            x, y = get_batch(train_data)
            logits = model(x)

            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                y.reshape(-1),
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0
            )
            optimizer.step()

            train_losses.append(loss.item())

        train_loss = sum(train_losses) / len(train_losses)
        val_loss = evaluate(val_data)

        val_perplexity = math.exp(min(val_loss, 20))

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_perplexity": val_perplexity,
        })

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val PPL: {val_perplexity:.2f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_count = 0

            # Save only to the separate v4 checkpoint.
            torch.save({
                "model_state_dict": model.state_dict(),
                "stoi": stoi,
                "itos": itos,
                "config": {
                    "vocab_size": len(chars),
                    "embed_size": EMBED_SIZE,
                    "num_layers": NUM_LAYERS,
                    "num_heads": NUM_HEADS,
                    "dropout": DROPOUT,
                    "max_length": BLOCK_SIZE,
                },
                "best_epoch": best_epoch,
                "best_val_loss": best_val_loss,
            }, CHECKPOINT_FILE)

        else:
            patience_count += 1

            if patience_count >= PATIENCE:
                print("Early stopping activated.")
                break

    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    HISTORY_FILE.write_text(
        json.dumps(history, indent=2),
        encoding="utf-8",
    )

    print("\nMiniGPT v4 training complete.")
    print(f"Best epoch: {best_epoch}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(
        "Best validation perplexity: "
        f"{math.exp(min(best_val_loss, 20)):.2f}"
    )
    print(f"Parameters: {parameter_count:,}")
    print(f"Training time: {time.time() - start_time:.2f} seconds")
    print(f"Checkpoint: {CHECKPOINT_FILE}")
    print(f"History: {HISTORY_FILE}")


if __name__ == "__main__":
    main()
