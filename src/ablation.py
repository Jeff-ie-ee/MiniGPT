import math
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import TextDataset
from tokenizer import CharacterTokenizer
from model import MiniGPT


# ============================================================
# 1. LOAD DATA
# ============================================================

with open("data/text.txt", "r", encoding="utf-8") as f:
    text = f.read()

split_index = int(0.9 * len(text))

train_text = text[:split_index]
val_text = text[split_index:]


# ============================================================
# 2. SETTINGS
# ============================================================

block_size = 16
batch_size = 2
epochs = 20
learning_rate = 0.001

embed_size = 64
num_layers = 1


# ============================================================
# 3. TOKENIZER
# ============================================================

tokenizer = CharacterTokenizer(text)

vocab_size = len(tokenizer.stoi)


# ============================================================
# 4. DATASETS
# ============================================================

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
# 5. EVALUATION
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
# 6. RUN ONE ABLATION
# ============================================================

def run_ablation(num_heads):

    print("\n" + "=" * 60)
    print(f"Attention Heads: {num_heads}")
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

        for x, y in train_loader:

            logits = model(x)

            loss = criterion(
                logits.reshape(-1, vocab_size),
                y.reshape(-1)
            )

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

    training_time = time.time() - start_time

    # Final training loss
    model.eval()

    train_total = 0.0

    with torch.no_grad():

        for x, y in train_loader:

            logits = model(x)

            loss = criterion(
                logits.reshape(-1, vocab_size),
                y.reshape(-1)
            )

            train_total += loss.item()

    train_loss = train_total / len(train_loader)

    # Validation loss
    val_loss = evaluate(
        model,
        criterion
    )

    train_ppl = math.exp(train_loss)
    val_ppl = math.exp(val_loss)

    print(f"Parameters      : {parameter_count:,}")
    print(f"Train Loss      : {train_loss:.4f}")
    print(f"Validation Loss : {val_loss:.4f}")
    print(f"Train PPL       : {train_ppl:.2f}")
    print(f"Validation PPL  : {val_ppl:.2f}")
    print(f"Time            : {training_time:.2f} sec")

    return {
        "heads": num_heads,
        "parameters": parameter_count,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "train_ppl": train_ppl,
        "val_ppl": val_ppl,
        "time": training_time
    }


# ============================================================
# 7. RUN ABLATIONS
# ============================================================

results = []

for heads in [1, 2, 4]:

    results.append(
        run_ablation(heads)
    )


# ============================================================
# 8. FINAL TABLE
# ============================================================

print("\n\n")
print("=" * 75)
print("ATTENTION HEAD ABLATION RESULTS")
print("=" * 75)

print(
    f"{'Heads':<10}"
    f"{'Params':>12}"
    f"{'Train Loss':>15}"
    f"{'Val Loss':>15}"
    f"{'Train PPL':>15}"
    f"{'Val PPL':>15}"
    f"{'Time':>10}"
)

print("-" * 75)

for result in results:

    print(
        f"{result['heads']:<10}"
        f"{result['parameters']:>12,}"
        f"{result['train_loss']:>15.4f}"
        f"{result['val_loss']:>15.4f}"
        f"{result['train_ppl']:>15.2f}"
        f"{result['val_ppl']:>15.2f}"
        f"{result['time']:>10.2f}"
    )