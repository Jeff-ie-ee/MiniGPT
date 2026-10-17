
from pathlib import Path
import json
import math
import random
import time

import torch
import torch.nn as nn

from src.model_v2 import MiniGPTv2


ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "data" / "text_expanded.txt"
CHECKPOINT_FILE = ROOT / "minigpt_v2.pt"
HISTORY_FILE = ROOT / "results" / "training_history_v2.json"

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
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}\n"
            "Run python expand_dataset.py first."
        )

    random.seed(SEED)
    torch.manual_seed(SEED)

    text = DATA_FILE.read_text(encoding="utf-8")
    if len(text) < 500:
        raise ValueError("Dataset is too small for this experiment.")

    # Stable character vocabulary.
    vocabulary = sorted(set(text))
    stoi = {char: index for index, char in enumerate(vocabulary)}
    itos = {index: char for char, index in stoi.items()}

    encoded = torch.tensor(
        [stoi[char] for char in text], dtype=torch.long
    )

    split = int(0.9 * len(encoded))
    train_data = encoded[:split]
    val_data = encoded[split:]

    if len(train_data) <= BLOCK_SIZE or len(val_data) <= BLOCK_SIZE:
        raise ValueError(
            "Training and validation data must both exceed block size."
        )

    def get_batch(source):
        starts = torch.randint(
            0, len(source) - BLOCK_SIZE, (BATCH_SIZE,)
        )
        x = torch.stack([source[i:i + BLOCK_SIZE] for i in starts])
        y = torch.stack([
            source[i + 1:i + BLOCK_SIZE + 1] for i in starts
        ])
        return x.to(DEVICE), y.to(DEVICE)

    model = MiniGPTv2(
        vocab_size=len(vocabulary),
        embed_size=EMBED_SIZE,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        dropout=DROPOUT,
        max_length=BLOCK_SIZE,
    ).to(DEVICE)

    parameters = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    criterion = nn.CrossEntropyLoss()

    def evaluate(source):
        model.eval()
        losses = []

        with torch.no_grad():
            for _ in range(EVAL_BATCHES):
                x, y = get_batch(source)
                logits = model(x)
                loss = criterion(
                    logits.reshape(-1, len(vocabulary)),
                    y.reshape(-1),
                )
                losses.append(loss.item())

        model.train()
        return sum(losses) / len(losses)

    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history = []
    start_time = time.time()

    print(f"Device: {DEVICE}")
    print(f"Dataset characters: {len(text):,}")
    print(f"Training characters: {len(train_data):,}")
    print(f"Validation characters: {len(val_data):,}")
    print(f"Vocabulary size: {len(vocabulary)}")
    print(f"Context length: {BLOCK_SIZE}")
    print(f"Transformer layers: {NUM_LAYERS}")
    print(f"Attention heads: {NUM_HEADS}")
    print(f"Model parameters: {parameters:,}")
    print(f"Training steps per epoch: {STEPS_PER_EPOCH}")
    print("\nTraining MiniGPT v2...\n")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        losses = []

        for _ in range(STEPS_PER_EPOCH):
            x, y = get_batch(train_data)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(
                logits.reshape(-1, len(vocabulary)),
                y.reshape(-1),
            )
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=1.0
            )
            optimizer.step()
            losses.append(loss.item())

        train_loss = sum(losses) / len(losses)
        val_loss = evaluate(val_data)

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_perplexity": math.exp(min(val_loss, 20)),
        }
        history.append(record)

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val PPL: {record['val_perplexity']:.2f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0

            checkpoint = {
                "model_state_dict": model.state_dict(),
                "tokenizer_stoi": stoi,
                "tokenizer_itos": itos,
                "vocab_size": len(vocabulary),
                "embed_size": EMBED_SIZE,
                "num_layers": NUM_LAYERS,
                "num_heads": NUM_HEADS,
                "dropout": DROPOUT,
                "max_length": BLOCK_SIZE,
                "forward_expansion": 4,
                "best_epoch": best_epoch,
                "val_loss": best_val_loss,
                "val_perplexity": math.exp(min(best_val_loss, 20)),
                "parameters": parameters,
                "dataset_file": str(DATA_FILE.name),
                "seed": SEED,
                "architecture": "MiniGPTv2",
            }
            torch.save(checkpoint, CHECKPOINT_FILE)
        else:
            epochs_without_improvement += 1

        HISTORY_FILE.write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )

        if epochs_without_improvement >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch}.")
            break

    print("\nMiniGPT v2 training complete.")
    print(f"Best epoch: {best_epoch}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best validation perplexity: {math.exp(min(best_val_loss, 20)):.2f}")
    print(f"Parameters: {parameters:,}")
    print(f"Training time: {time.time() - start_time:.2f} seconds")
    print(f"Checkpoint: {CHECKPOINT_FILE}")
    print(f"History: {HISTORY_FILE}")


if __name__ == "__main__":
    main()
