
from pathlib import Path

# Project paths
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

# Input and output files
BACKUP_FILE = DATA_DIR / "text_backup.txt"
ORIGINAL_FILE = DATA_DIR / "text.txt"
OUTPUT_FILE = DATA_DIR / "text_expanded.txt"

EXTRA_TEXT = """
Artificial Intelligence and Machine Learning

Artificial intelligence enables computers to perform tasks such
as learning, reasoning, prediction, and pattern recognition.

Machine learning is a branch of artificial intelligence. It uses
data to learn patterns and make predictions. A machine learning
workflow includes data collection, preprocessing, model training,
evaluation, and deployment.

Supervised learning uses labeled data. Classification predicts
categories, while regression predicts numerical values.

Unsupervised learning discovers patterns in unlabeled data.
Clustering groups similar examples. Dimensionality reduction
represents data with fewer features.

Training data helps a model learn. Validation data helps select
model settings. Test data measures performance on unseen examples.
Data leakage happens when information that should not be available
during prediction influences the training process.

Neural Networks

A neural network contains connected computational units called
neurons. A basic network has input, hidden, and output layers.
Weights and biases are learned during training.

Activation functions introduce nonlinearity. Common examples
include ReLU, sigmoid, and tanh.

Backpropagation calculates gradients of the loss function.
An optimizer uses those gradients to update model parameters.
The learning rate controls the size of parameter updates.

Overfitting occurs when a model learns training examples too
closely and performs poorly on unseen data. Dropout, weight decay,
early stopping, and better training data may help reduce it.

Natural Language Processing

Natural language processing enables computers to process text.
Applications include text classification, translation,
summarization, question answering, and text generation.

Tokenization divides text into smaller units called tokens.
A character-level tokenizer uses individual characters as tokens.
An embedding converts a token identifier into a vector.

A language model learns patterns in sequences of tokens.
An autoregressive language model predicts the next token using
the preceding context. It can generate text by repeatedly
predicting and adding new tokens.

Transformer Architecture

A Transformer uses attention to model relationships between
tokens. Self-attention combines information from different
positions in a sequence.

Attention uses queries, keys, and values. Query and key
similarities determine attention weights. These weights combine
value vectors into contextual representations.

Multi-head attention uses several attention heads. Each head
can learn different relationships between tokens.

Positional information helps the model represent token order.
Residual connections help information flow through the network.
Layer normalization helps stabilize training.

A causal mask prevents a language model from using future tokens
when predicting the next token.

Model Training and Evaluation

Cross-entropy loss measures how well predicted probabilities
match the correct next token. Lower loss generally indicates
better predictions on the same evaluation dataset.

Perplexity is the exponential of average cross-entropy loss
when natural logarithms are used. Lower perplexity generally
indicates better next-token prediction on the same dataset.

An epoch is one pass through the training dataset. Batch size
controls how many examples are processed in an update.
The learning rate controls the magnitude of optimizer updates.

Training loss and validation loss should both be monitored.
A model with low training loss but high validation loss may
be overfitting.

Text Generation

Greedy decoding selects the most probable next token.
Sampling selects a token using its predicted probability.
Temperature changes the sharpness of the probability distribution.
Top-k sampling restricts choices to the k most probable tokens.

A small language model may repeat phrases or produce grammatical
errors. Results depend on training data, model size, optimization,
and text-generation settings.

Computer Vision

Computer vision enables computers to analyze images and videos.
A convolutional neural network uses filters to extract features.
Stride controls filter movement, and padding controls border
handling.

A feature map contains the output of a filter. Pooling can reduce
spatial dimensions. Fully connected layers combine features
to produce predictions.

Responsible Machine Learning

Machine learning systems should be evaluated for accuracy,
reliability, fairness, privacy, and security. Training data
should be representative of the intended application.

A small educational language model is useful for learning
tokenization, attention, optimization, and text generation.
It is not equivalent to a large-scale language model.
"""


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Prefer the backup so repeated runs do not duplicate extra text.
    if BACKUP_FILE.exists():
        source_file = BACKUP_FILE
    elif ORIGINAL_FILE.exists():
        source_file = ORIGINAL_FILE
    else:
        raise FileNotFoundError(
            "Dataset not found.\n"
            f"Expected either:\n{BACKUP_FILE}\n"
            f"or:\n{ORIGINAL_FILE}"
        )

    original_text = source_file.read_text(encoding="utf-8").strip()

    if not original_text:
        raise ValueError(f"The source dataset is empty: {source_file}")

    expanded_text = original_text + "\n\n" + EXTRA_TEXT.strip() + "\n"
    OUTPUT_FILE.write_text(expanded_text, encoding="utf-8")

    print("\nDataset created successfully!")
    print(f"Source: {source_file}")
    print(f"Original characters: {len(original_text):,}")
    print(f"Added characters: {len(EXTRA_TEXT):,}")
    print(f"Total characters: {len(expanded_text):,}")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
