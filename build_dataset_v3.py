
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

# Prefer the original backup to avoid repeatedly appending prior additions.
SOURCE_FILE = DATA_DIR / "text_backup.txt"
OUTPUT_FILE = DATA_DIR / "text_v3.txt"


EXTRA_TEXT = r"""
CHAPTER: INTRODUCTION TO ARTIFICIAL INTELLIGENCE

Artificial intelligence is a field of computer science concerned with
building systems that perform tasks requiring reasoning, learning,
planning, perception, or language processing. An intelligent system
receives information, processes it, and produces an output or action.
Different AI systems use different representations and algorithms.

Machine learning is a subfield of artificial intelligence. Instead of
writing a separate rule for every possible situation, developers train
a model using examples. The model adjusts internal parameters to learn
patterns that can be useful when processing new inputs.

Consider an email classifier. Its input may contain the subject and
body of an email. Its output may be the label spam or not spam. During
training, the system receives examples with known labels. After
training, it predicts a label for an email it has not seen before.
Its predictions should be evaluated on independent examples.

A machine learning project involves more than selecting an algorithm.
A reliable workflow defines the problem, collects relevant data,
checks data quality, creates training and evaluation sets, chooses a
baseline, trains a model, measures performance, examines errors, and
documents limitations. A successful experiment should be reproducible.

CHAPTER: TYPES OF MACHINE LEARNING

Supervised learning uses labeled examples. Each example contains
features and a target. Classification predicts a discrete category.
Regression predicts a numerical value. A house price model is a
regression problem, while an image category model is a classification
problem.

Unsupervised learning studies data without predefined target labels.
Clustering attempts to group similar examples. Dimensionality
reduction represents observations using fewer dimensions. These
methods can help explore a dataset, but clusters do not automatically
correspond to meaningful real-world categories.

Reinforcement learning studies an agent that interacts with an
environment. The agent observes a state, chooses an action, and
receives feedback in the form of a reward. Its objective is to learn a
policy that achieves good long-term outcomes. The quality of the
reward design can strongly affect the learned behavior.

A baseline model provides a reference point for later experiments.
A complicated model is not automatically better than a simple one.
If logistic regression solves a task reliably, a much larger neural
network may not justify its extra computation and complexity.

CHAPTER: DATA PREPROCESSING AND DATA QUALITY

Raw data can contain missing values, duplicated records, inconsistent
units, spelling differences, incorrect labels, and irrelevant fields.
Preprocessing identifies and handles these problems. The correct
approach depends on the source, meaning, and intended use of the data.

Features are the input variables used by a model. Feature engineering
creates or transforms variables to represent useful information.
Normalization and standardization can help algorithms that are
sensitive to feature scales. Categorical variables may need to be
encoded before numerical model operations.

Data leakage occurs when information unavailable at prediction time
influences training or model selection. For example, fitting a
normalization transform on the complete dataset before splitting it
can leak information from the validation set. Fit preprocessing
operations on training data and apply the fitted operations to other
splits.

A dataset should reflect the intended deployment setting. If a model
will predict future events, a time-based split may be more appropriate
than a random split. If multiple records belong to the same person,
keeping related records in one split may reduce leakage.

CHAPTER: MODEL TRAINING AND OPTIMIZATION

A model contains parameters that are learned from data. Hyperparameters
are settings chosen by the developer, such as learning rate, batch
size, number of layers, context length, and dropout probability.

A loss function measures the difference between a model's predictions
and its training targets. During optimization, the model computes
gradients and updates parameters to reduce the loss. A lower training
loss is useful, but it does not guarantee better predictions on new
examples.

Gradient descent updates parameters in a direction related to the
negative gradient of the loss. The learning rate determines the size
of each update. If the learning rate is too high, optimization can be
unstable. If it is too low, learning may be slow.

An epoch represents a pass through the training examples. A batch is
a group of examples used to calculate an optimization step. Small
batches can introduce noisy gradient estimates. Larger batches use
more memory and may require different optimization settings.

Adam and AdamW are popular optimizers. Adam adapts update magnitudes
using estimates of gradient statistics. AdamW separates weight decay
from the gradient-based update. Gradient clipping limits large
gradient norms and can help stabilize training.

CHAPTER: OVERFITTING AND GENERALIZATION

Generalization is the ability to make useful predictions on examples
not used to fit the model. Overfitting occurs when a model learns
training-specific details that do not transfer well to new data.
Underfitting occurs when a model cannot represent the patterns needed
to perform the task well.

Training loss measures performance on training examples. Validation
loss measures performance on a separate set used for model selection.
Test data should be reserved for final evaluation. Repeatedly tuning a
model against the test set makes that set part of the development
process and weakens the final evaluation.

Dropout randomly disables selected activations during training.
Weight decay discourages excessively large parameter values.
Early stopping ends training when validation performance no longer
improves. More representative training data can also improve
generalization when it captures relevant variation.

Learning curves help reveal model behavior. If both training and
validation losses remain high, the model may be underfitting. If
training loss falls while validation loss rises, overfitting may be
occurring. These patterns are clues rather than absolute diagnoses.

CHAPTER: NEURAL NETWORK FOUNDATIONS

An artificial neural network contains layers of computational units.
An input layer receives a representation of the data. Hidden layers
transform that representation. An output layer produces predictions
for the target task.

Each neuron combines input values with learned weights and a bias.
An activation function transforms the resulting value. ReLU returns
zero for negative inputs and passes positive inputs through. Sigmoid
maps values into a range between zero and one. Tanh maps values into
a range between negative one and one.

Backpropagation applies the chain rule to calculate derivatives of
the loss with respect to model parameters. The optimizer uses these
gradients to update the weights. Efficient automatic differentiation
makes it practical to train networks with many parameters.

A convolutional neural network is often used for images. Convolution
filters process local image regions and produce feature maps. Stride
controls how far a filter moves at each step. Padding determines how
the operation handles image boundaries. Pooling can reduce spatial
dimensions.

CHAPTER: NATURAL LANGUAGE PROCESSING

Natural language processing enables computer systems to process
human language. Applications include sentiment analysis, document
classification, translation, information extraction, summarization,
question answering, and text generation.

Text must be converted into a representation the model can process.
Tokenization divides text into tokens. A character-level tokenizer
uses individual characters. A word-level tokenizer uses whole words.
A subword tokenizer uses common pieces of words and can represent
many uncommon words through combinations of those pieces.

A vocabulary maps tokens to numerical identifiers. An embedding maps
each identifier to a learned vector. The embedding represents a token
in a continuous vector space. During training, embeddings can acquire
useful patterns from the surrounding context.

A language model assigns probabilities to sequences of tokens.
An autoregressive language model predicts the next token based on
previous tokens. Generation repeats this operation, appending each
predicted token to the existing sequence.

A character-level language model predicts one character at a time.
It can represent any character in its vocabulary, but producing a
word requires a sequence of correct character predictions. Small
character-level models often struggle with spelling and long-range
sentence structure.

CHAPTER: TRANSFORMER ARCHITECTURE

The Transformer is a neural network architecture built around
attention. Attention allows a token representation to combine
information from other permitted positions in a sequence. Transformers
are widely used in language modeling and many other tasks.

Self-attention uses queries, keys, and values. Similarities between
queries and keys determine attention weights. The weights are applied
to value vectors to produce contextual representations. Different
attention heads can learn different relationships between tokens.

A causal language model must not look at future target tokens while
predicting the next token. A causal mask prevents attention to future
positions. This restriction ensures that the prediction at a position
depends only on the allowed context.

Positional embeddings provide information about sequence order.
Residual connections add a block's input to its output. Layer
normalization helps stabilize the scale of intermediate activations.
Feedforward layers apply learned transformations to each position.

A Transformer block commonly combines attention and a feedforward
network. Multiple blocks allow representations to be transformed
repeatedly. Increasing depth and width adds capacity but also increases
the amount of data and computation needed for effective training.

CHAPTER: LANGUAGE MODEL EVALUATION

Cross-entropy loss measures how much probability a model assigns to
the correct target token. When natural logarithms are used, perplexity
is the exponential of average cross-entropy loss. Lower perplexity
usually indicates better next-token prediction on the same evaluation
examples with the same tokenization.

Metrics must be compared carefully. Two models should use the same
evaluation text, target positions, vocabulary, and loss calculation.
A model evaluated on an easier dataset can have a lower loss without
being better overall.

Generated samples provide another type of evidence. Review whether
the output is relevant to the prompt, whether sentences are readable,
whether words are spelled correctly, and whether the model repeats
itself. A single attractive sample does not establish general quality.

Temperature changes the sharpness of the next-token distribution.
Lower temperatures usually favor more probable tokens. Higher
temperatures usually increase variation. Top-k sampling restricts
the available choices to a set of high-probability tokens. Sampling
settings can change style, but they cannot supply missing knowledge.

CHAPTER: EXPERIMENT DESIGN AND REPRODUCIBILITY

A reliable experiment records the dataset version, data split, model
configuration, random seed, optimizer, learning rate, batch size,
training duration, and evaluation metrics. Keeping these details
allows another person to reproduce the experiment.

Change one major factor at a time when possible. If model size,
dataset, learning rate, and training procedure all change together,
it becomes difficult to identify which change affected the results.

Save checkpoints separately. Preserve a known working model while
testing a candidate. Promote a new checkpoint only after it loads
correctly, passes generation tests, and performs adequately on the
chosen evaluation set.

A project report should include limitations and failure cases.
A small educational Transformer is useful for demonstrating
tokenization, embeddings, causal attention, optimization, and
generation. It should not be presented as equivalent to a large
pretrained language model.

CHAPTER: RESPONSIBLE AI

Machine learning systems can inherit biases from their training data.
A high average score may hide poor performance on specific examples
or groups. Evaluation should consider the intended use and relevant
failure modes.

Data privacy matters when collecting, storing, or processing personal
information. Developers should use appropriate data permissions and
protect sensitive records. Model outputs should be checked before
they are used for high-impact decisions.

A deployed system can degrade when its input distribution changes.
Monitoring, testing, documentation, and periodic evaluation help
identify problems. Responsible development includes communicating
what a model can do, what it cannot do, and where it may fail.
"""


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if SOURCE_FILE.exists():
        source = SOURCE_FILE
    elif (DATA_DIR / "text.txt").exists():
        source = DATA_DIR / "text.txt"
    elif (DATA_DIR / "text_expanded.txt").exists():
        source = DATA_DIR / "text_expanded.txt"
    else:
        raise FileNotFoundError(
            "Could not find a dataset. Expected one of:\n"
            "data/text_backup.txt\n"
            "data/text.txt\n"
            "data/text_expanded.txt"
        )

    original_text = source.read_text(encoding="utf-8").strip()
    if not original_text:
        raise ValueError(f"Source file is empty: {source}")

    # Do not append EXTRA_TEXT twice if this script is run again:
    # it always reads the source file and writes a separate output.
    combined = original_text + "\n\n" + EXTRA_TEXT.strip() + "\n"
    OUTPUT_FILE.write_text(combined, encoding="utf-8")

    print("\nMiniGPT v3 dataset created.")
    print(f"Source file: {source}")
    print(f"Original characters: {len(original_text):,}")
    print(f"New educational characters: {len(EXTRA_TEXT):,}")
    print(f"Total characters: {len(combined):,}")
    print(f"Unique characters: {len(set(combined))}")
    print(f"Output file: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
