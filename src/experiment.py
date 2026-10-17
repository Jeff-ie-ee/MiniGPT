import math
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import TextDataset
from tokenizer import CharacterTokenizer
from model import MiniGPT


# ============================================================
# SETTINGS
# ============================================================

with open("data/text.txt", "r", encoding="utf-8") as f:
    text = f.read()

split_index = int(0.9 * len(text))

train_text = text[:split_index]
val_text = text[split_index:]

block_size = 16
batch_size = 2
epochs = 20
learning_rate = 0.001


# ============================================================
# SHARED TOKENIZER
# ============================================================

tokenizer = CharacterTokenizer(text)

vocab_size = len(tokenizer.stoi)

train_dataset = TextDataset(
    train_text,
    tokenizer,
    block_size
)

val_dataset = TextDataset(
    val_text,
    tokenizer,
    block_size
)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False
)


# ============================================================
# EVALUATION
# ============================================================

def evaluate(model, criterion):

    model.eval()

    total_loss = 0.0

    with torch.no_grad():

        for x, y in val_loader:

            logits = model(x)

            loss = criterion(
                logits.reshape(-1, vocab_size),
                y.reshape(-1)
            )

            total_loss += loss.item()

    return total_loss / len(val_loader)


# ============================================================
# TRAIN ONE MODEL
# ============================================================

def run_experiment(name, embed_size, num_layers, num_heads):

    print("\n" + "=" * 60)
    print(f"Experiment: {name}")
    print("=" * 60)

    model = MiniGPT(
        vocab_size=vocab_size,
        embed_size=embed_size,
        num_layers=num_layers,
        num_heads=num_heads,
        forward_expansion=4,
        dropout=0.1,
        max_length=block_size
    )

    parameter_count = sum(
        p.numel()
        for p in model.parameters()
    )

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate
    )

    start_time = time.time()

    for epoch in range(epochs):

        model.train()

        total_loss = 0.0

        for x, y in train_loader:

            logits = model(x)

            loss = criterion(
                logits.reshape(-1, vocab_size),
                y.reshape(-1)
            )

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

    training_time = time.time() - start_time

    train_loss = total_loss / len(train_loader)

    val_loss = evaluate(
        model,
        criterion
    )

    train_ppl = math.exp(train_loss)
    val_ppl = math.exp(val_loss)

    print(f"Parameters       : {parameter_count:,}")
    print(f"Training Loss    : {train_loss:.4f}")
    print(f"Validation Loss  : {val_loss:.4f}")
    print(f"Training PPL     : {train_ppl:.2f}")
    print(f"Validation PPL   : {val_ppl:.2f}")
    print(f"Training Time    : {training_time:.2f} seconds")

    return {
        "name": name,
        "parameters": parameter_count,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "train_ppl": train_ppl,
        "val_ppl": val_ppl,
        "time": training_time
    }


# ============================================================
# RUN EXPERIMENTS
# ============================================================

results = []

results.append(
    run_experiment(
        "Small",
        embed_size=32,
        num_layers=1,
        num_heads=2
    )
)

results.append(
    run_experiment(
        "Medium",
        embed_size=64,
        num_layers=1,
        num_heads=2
    )
)

results.append(
    run_experiment(
        "Large",
        embed_size=64,
        num_layers=2,
        num_heads=4
    )
)


# ============================================================
# FINAL RESULTS
# ============================================================

print("\n\n")
print("=" * 80)
print("FINAL EXPERIMENT RESULTS")
print("=" * 80)

print(
    f"{'Model':<10}"
    f"{'Params':>12}"
    f"{'Train Loss':>15}"
    f"{'Val Loss':>15}"
    f"{'Train PPL':>15}"
    f"{'Val PPL':>15}"
    f"{'Time(s)':>12}"
)

print("-" * 80)

for result in results:

    print(
        f"{result['name']:<10}"
        f"{result['parameters']:>12,}"
        f"{result['train_loss']:>15.4f}"
        f"{result['val_loss']:>15.4f}"
        f"{result['train_ppl']:>15.2f}"
        f"{result['val_ppl']:>15.2f}"
        f"{result['time']:>12.2f}"
    )