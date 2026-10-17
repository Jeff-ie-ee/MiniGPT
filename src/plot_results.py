import json
import os

import matplotlib.pyplot as plt


# Load training history
history_path = "results/training_history.json"

if not os.path.exists(history_path):
    raise FileNotFoundError(
        "results/training_history.json not found. "
        "Run python src/train.py first."
    )


with open(
    history_path,
    "r",
    encoding="utf-8"
) as f:
    history = json.load(f)


train_loss = history["train_loss"]
val_loss = history["val_loss"]

epochs = range(1, len(train_loss) + 1)


# Create loss graph
plt.figure(figsize=(9, 5))

plt.plot(
    epochs,
    train_loss,
    marker="o",
    label="Training Loss"
)

plt.plot(
    epochs,
    val_loss,
    marker="o",
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("MiniGPT Training vs Validation Loss")

plt.legend()
plt.grid(True)
plt.tight_layout()


# Save graph
os.makedirs("results", exist_ok=True)

output_path = "results/loss_curve.png"

plt.savefig(
    output_path,
    dpi=200
)

print(f"Loss graph saved to: {output_path}")

