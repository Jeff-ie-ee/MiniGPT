
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import TextDataset
from tokenizer import CharacterTokenizer
from model import MiniGPT


# ============================================================
# 1. PROJECT PATHS AND REPRODUCIBILITY
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "text.txt"
RESULTS_DIR = BASE_DIR / "results"

# Do not overwrite the working production checkpoint yet.
CANDIDATE_PATH = BASE_DIR / "minigpt_candidate.pt"
HISTORY_PATH = RESULTS_DIR / "training_history_candidate.json"

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cpu")
print("Device:", DEVICE)


# ============================================================
# 2. LOAD AND VALIDATE DATA
# ============================================================

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}"
    )

text = DATA_PATH.read_text(encoding="utf-8")

if len(text) < 1000:
    raise ValueError(
        "Dataset is too small. Add more text to data/text.txt."
    )

if not text.strip():
    raise ValueError("Dataset contains no usable text.")

print("Total characters:", len(text))


# ============================================================
# 3. TRAIN / VALIDATION SPLIT
# ============================================================

split_index = int(0.90 * len(text))

train_text = text[:split_index]
val_text = text[split_index:]

print("Training characters:", len(train_text))
print("Validation characters:", len(val_text))


# ============================================================
# 4. CONFIGURATION
# Keep the architecture compatible with the existing API.
# ============================================================

BLOCK_SIZE = 32
BATCH_SIZE = 2
EPOCHS = 30
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0
EARLY_STOPPING_PATIENCE = 8

EMBED_SIZE = 64
NUM_LAYERS = 1
NUM_HEADS = 2
FORWARD_EXPANSION = 4
DROPOUT = 0.1


# ============================================================
# 5. TOKENIZER
# ============================================================

tokenizer = CharacterTokenizer(text)
vocab_size = len(tokenizer.stoi)

print("Vocabulary size:", vocab_size)


# ============================================================
# 6. DATASETS AND LOADERS
# ============================================================

train_dataset = TextDataset(
    train_text, tokenizer, BLOCK_SIZE
)

val_dataset = TextDataset(
    val_text, tokenizer, BLOCK_SIZE
)

if len(train_dataset) == 0 or len(val_dataset) == 0:
    raise ValueError(
        "Training or validation dataset is empty. "
        "Check TextDataset and BLOCK_SIZE."
    )

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print("Training batches:", len(train_loader))
print("Validation batches:", len(val_loader))


# ============================================================
# 7. MODEL
# ============================================================

model = MiniGPT(
    vocab_size=vocab_size,
    embed_size=EMBED_SIZE,
    num_layers=NUM_LAYERS,
    num_heads=NUM_HEADS,
    forward_expansion=FORWARD_EXPANSION,
    dropout=DROPOUT,
    max_length=BLOCK_SIZE
).to(DEVICE)

total_parameters = sum(
    p.numel() for p in model.parameters()
)

print(f"Model parameters: {total_parameters:,}")


# ============================================================
# 8. LOSS, OPTIMIZER, SCHEDULER
# ============================================================

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=2
)


# ============================================================
# 9. EVALUATION HELPERS
# Accumulate token-weighted loss instead of averaging batches.
# ============================================================

def run_epoch(training):
    model.train(training)

    total_loss = 0.0
    total_tokens = 0

    context = torch.enable_grad if training else torch.no_grad

    with context():
        for x, y in (
            train_loader if training else val_loader
        ):
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            if training:
                optimizer.zero_grad(set_to_none=True)

            logits = model(x)

            loss = criterion(
                logits.reshape(-1, vocab_size),
                y.reshape(-1)
            )

            if training:
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    GRAD_CLIP
                )

                optimizer.step()

            token_count = y.numel()
            total_loss += loss.item() * token_count
            total_tokens += token_count

    if total_tokens == 0:
        raise RuntimeError("No tokens processed in this epoch.")

    return total_loss / total_tokens


def perplexity(loss):
    return math.exp(min(loss, 20))


# ============================================================
# 10. TRAINING HISTORY
# ============================================================

history = {
    "dataset_characters": len(text),
    "training_characters": len(train_text),
    "validation_characters": len(val_text),
    "vocabulary_size": vocab_size,
    "seed": SEED,
    "block_size": BLOCK_SIZE,
    "batch_size": BATCH_SIZE,
    "configured_epochs": EPOCHS,
    "learning_rate": LEARNING_RATE,
    "weight_decay": WEIGHT_DECAY,
    "model_parameters": total_parameters,
    "train_loss": [],
    "val_loss": [],
    "train_perplexity": [],
    "val_perplexity": [],
    "learning_rate_history": [],
    "best_epoch": None,
    "best_val_loss": None
}


# ============================================================
# 11. TRAIN
# ============================================================

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

best_val_loss = float("inf")
best_epoch = 0
epochs_without_improvement = 0

start_time = time.time()

print("\n========== STARTING MINIGPT TRAINING ==========\n")

for epoch in range(1, EPOCHS + 1):
    train_loss = run_epoch(training=True)
    val_loss = run_epoch(training=False)

    scheduler.step(val_loss)

    train_ppl = perplexity(train_loss)
    val_ppl = perplexity(val_loss)
    current_lr = optimizer.param_groups[0]["lr"]

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["train_perplexity"].append(train_ppl)
    history["val_perplexity"].append(val_ppl)
    history["learning_rate_history"].append(current_lr)

    improved = val_loss < best_val_loss

    if improved:
        best_val_loss = val_loss
        best_epoch = epoch
        epochs_without_improvement = 0

        # Keep the API-compatible checkpoint structure.
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "tokenizer_stoi": tokenizer.stoi,
            "tokenizer_itos": tokenizer.itos,
            "vocab_size": vocab_size,
            "embed_size": EMBED_SIZE,
            "num_layers": NUM_LAYERS,
            "num_heads": NUM_HEADS,
            "forward_expansion": FORWARD_EXPANSION,
            "dropout": DROPOUT,
            "max_length": BLOCK_SIZE,
            "dataset_characters": len(text),
            "training_characters": len(train_text),
            "validation_characters": len(val_text),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_perplexity": train_ppl,
            "val_perplexity": val_ppl,
            "parameters": total_parameters,
            "seed": SEED,
            "best_epoch": best_epoch
        }

        # Save to a separate candidate file.
        torch.save(checkpoint, CANDIDATE_PATH)

    else:
        epochs_without_improvement += 1

    print(
        f"Epoch [{epoch:02d}/{EPOCHS}] "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Train PPL: {train_ppl:.2f} | "
        f"Val PPL: {val_ppl:.2f} | "
        f"LR: {current_lr:.6f}"
        + (" | Best model saved" if improved else "")
    )

    history["best_epoch"] = best_epoch
    history["best_val_loss"] = best_val_loss

    if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
        print(
            f"\nEarly stopping: validation loss did not improve "
            f"for {EARLY_STOPPING_PATIENCE} epochs."
        )
        break


# ============================================================
# 12. SAVE EXPERIMENT HISTORY
# ============================================================

history["epochs_completed"] = len(history["train_loss"])
history["training_seconds"] = round(time.time() - start_time, 2)
history["best_val_perplexity"] = perplexity(best_val_loss)

HISTORY_PATH.write_text(
    json.dumps(history, indent=2),
    encoding="utf-8"
)


# ============================================================
# 13. SUMMARY
# ============================================================

print("\n========== TRAINING COMPLETE ==========")
print("Epochs completed:", history["epochs_completed"])
print("Best epoch:", best_epoch)
print(f"Best validation loss: {best_val_loss:.4f}")
print(f"Best validation perplexity: {perplexity(best_val_loss):.2f}")
print(f"Parameters: {total_parameters:,}")
print(f"Training time: {history['training_seconds']} seconds")

print("\nSaved candidate model:")
print(CANDIDATE_PATH)

print("\nSaved training history:")
print(HISTORY_PATH)

print("\nYour existing minigpt.pt has not been overwritten.")
