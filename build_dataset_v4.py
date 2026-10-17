
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUT_FILE = DATA_DIR / "text_v4.txt"

# Prefer the original source to avoid repeatedly appending generated text.
SOURCE_CANDIDATES = [
    DATA_DIR / "text_backup.txt",
    DATA_DIR / "text.txt",
    DATA_DIR / "text_expanded.txt",
    DATA_DIR / "text_v3.txt",
]

EXTRA_TEXT = r"""
Artificial intelligence: practical applications

An artificial intelligence application begins with a clearly defined
problem. A developer identifies the input, the desired output, and the
conditions under which the system will be used. The developer then
selects data and a suitable method. The completed system must be tested
on examples that were not used to fit its parameters.

A recommendation system estimates which items may be useful to a
person. It can use item properties, user preferences, and patterns in
previous interactions. A recommendation is a prediction rather than a
guarantee. If historical data reflects an existing bias, the system
may reproduce that bias. Evaluation should therefore consider both
prediction quality and the consequences of the recommendations.

A computer vision system processes images or video. It may classify
an image, detect objects, segment regions, or estimate movement.
Image quality, lighting, camera position, and background can affect
its performance. A system tested only on clear photographs may fail
when images are dark, blurred, or taken from an unfamiliar viewpoint.

An AI assistant processes a user's request and produces a response.
A language model can generate a plausible answer even when its
information is incomplete. Applications should make uncertainty
visible and verify important claims where possible. A generated
answer should not automatically be treated as a verified fact.

Understanding data

A dataset is a collection of observations used to investigate a
question or train a model. Each observation contains one or more
fields. A table may contain rows for examples and columns for
attributes. The meaning, units, and permitted values of each field
should be documented.

Data cleaning includes identifying invalid values, resolving
inconsistent formats, checking duplicates, and deciding how to handle
missing information. Deleting every row with a missing value can
introduce bias if the missingness is systematic. Imputation can be
useful, but its assumptions should be documented and evaluated.

A training set is used to fit model parameters. A validation set helps
select settings and compare candidate models. A final test set
provides an estimate of performance after development decisions are
complete. These sets should be separated before fitting transformations
that learn information from the data.

A random split is not appropriate for every problem. Time series
forecasting should respect the order of time. Medical records or user
events may contain multiple observations from the same individual.
Related observations should be grouped carefully to reduce leakage
between training and evaluation data.

Machine learning algorithms

Linear regression estimates a numerical target using a weighted
combination of input features. It is a useful baseline because its
predictions are relatively easy to inspect. A linear model may not
capture a nonlinear relationship unless suitable features are added.

Logistic regression is commonly used for classification. Despite its
name, it predicts class probabilities rather than ordinary continuous
regression targets. A decision threshold converts the probability into
a class prediction. The threshold should reflect the requirements of
the application.

A decision tree repeatedly divides examples according to feature
values. Small trees can be interpreted by following a path from the
root to a leaf. Deep trees can fit the training data closely and may
generalize poorly. Limiting tree depth or requiring more examples per
leaf can help control complexity.

A random forest combines predictions from multiple decision trees.
Different training samples and feature selections create variation
between the trees. Aggregating their predictions often produces a
more stable result than relying on a single tree.

A support vector machine seeks a decision boundary with a large
margin between classes. Its behavior depends on the kernel, regular-
ization, and feature representation. Feature scaling is often
important when numerical variables have very different ranges.

Neural networks and optimization

A neural network transforms input vectors through learned layers.
Weights determine how strongly inputs influence later computations.
Biases allow a neuron to shift its response. Activation functions
introduce nonlinear behavior so that a network can represent more
complex relationships than a single linear transformation.

The forward pass computes predictions from the current parameters.
A loss function compares those predictions with the target values.
Backpropagation calculates gradients of the loss with respect to the
parameters. An optimizer then uses those gradients to update the
parameters. This cycle repeats over many batches.

Training loss alone is not enough to assess a model. A network can
memorize examples when its capacity is large relative to the available
data. Validation measurements, regularization, careful data splits,
and representative examples help assess generalization.

A random seed can make certain randomized operations repeatable,
but it does not guarantee identical results across every device,
library version, or hardware configuration. Reproducibility also
requires recording code versions, data sources, and training settings.

Language modeling

A language model estimates the probability of a token given earlier
tokens. In an autoregressive model, the next-token distribution is
conditioned on the available context. Generation samples or selects a
token from this distribution and adds it to the context.

Character-level modeling uses characters as tokens. It has a simple
vocabulary but must learn spelling and word boundaries from character
sequences. Word-level modeling uses larger units but needs a strategy
for words that were not present in the vocabulary. Subword methods
offer a compromise by representing words as reusable pieces.

The context window is the amount of preceding text available to a
prediction. A short window limits the relationships the model can
directly inspect. A longer window requires more computation and memory,
especially when attention is used.

Causal attention restricts each position to information from that
position and earlier positions. Without this restriction, a training
position could access the future target it is supposed to predict.
Correct masking is therefore essential for autoregressive training.

The Transformer uses attention to combine information from token
representations. Queries and keys determine compatibility scores.
The scores are converted into weights and used to combine values.
Multiple heads can learn different patterns of relationships. A
feedforward network then transforms the representation at each
position.

Embeddings convert discrete token identifiers into vectors. Positional
information helps distinguish different positions in a sequence.
Residual connections provide paths for information and gradients.
Layer normalization helps stabilize intermediate representations.
These components work together to support the learning process.

Evaluating text generation

Cross-entropy loss penalizes a model when it assigns low probability
to the correct next token. Perplexity is the exponential of average
cross-entropy when natural logarithms are used. It is useful for
comparing language models evaluated on the same targets and with the
same tokenization and loss calculation.

Generated text should also be inspected. A useful review checks
grammar, spelling, relevance, repetition, and whether the response
continues the prompt appropriately. A few samples cannot establish
that a model performs well on every type of input.

Temperature changes the probability distribution used for sampling.
A lower temperature concentrates probability on likely tokens.
A higher temperature produces more varied choices. Top-k sampling
limits the choices to the k most likely tokens. These methods affect
generation style but do not replace good training data.

An experiment should change a limited number of factors at a time.
If a new model changes the dataset, vocabulary, architecture, and
optimizer simultaneously, it becomes difficult to explain a change
in performance. Controlled experiments make conclusions more useful.

Software engineering for AI

A machine learning project should separate data preparation, model
definition, training, evaluation, and inference. Modular code makes
each part easier to test. Configuration values should be documented
rather than scattered throughout the project.

A checkpoint should include the model parameters and the configuration
needed to reconstruct the architecture. Tokenizer mappings must also
be preserved when token identifiers are used. A checkpoint trained
with one vocabulary cannot safely be loaded with an unrelated
vocabulary merely because the model dimensions happen to match.

Automated tests can verify that model outputs have the expected shape,
that loss values are finite, and that generation produces valid
tokens. A smoke test can load a checkpoint and generate a short
sequence before the model is connected to an API.

An API should validate its inputs and return useful error messages.
Long generation requests may need a maximum token limit. Logging can
help diagnose failures, but logs should not unnecessarily contain
private user information.

Performance should be measured rather than assumed. More parameters
may improve capacity, but they also require additional computation.
A small model with suitable data and a clear task can be a useful
educational project even when its generated prose is limited.

Project documentation should explain the objective, architecture,
dataset, training procedure, evaluation, and known limitations.
Results should be reported accurately. An educational model should
not be described as a production-ready assistant without appropriate
testing and evidence.

Ethics and reliability

AI systems can make mistakes because of limited training data,
distribution changes, ambiguous inputs, and limitations in their
design. A responsible developer tests common cases and failure cases.
When a prediction affects an important decision, additional review
may be necessary.

Privacy should be considered when gathering training examples.
Data should be collected and processed with appropriate permission.
Sensitive information should be protected, and unnecessary records
should not be retained.

A model that works well on one dataset may perform poorly in a
different environment. Monitoring and periodic evaluation can reveal
changes in input patterns or output quality. Documentation should
explain intended uses, limitations, and conditions that require
human oversight.
"""


def normalize_paragraph(paragraph):
    return re.sub(r"\s+", " ", paragraph).strip()


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    source_file = next(
        (path for path in SOURCE_CANDIDATES if path.exists()),
        None,
    )

    if source_file is None:
        raise FileNotFoundError(
            "No source dataset found. Add data/text.txt or "
            "data/text_backup.txt first."
        )

    source_text = source_file.read_text(encoding="utf-8").strip()

    if not source_text:
        raise ValueError(f"The source dataset is empty: {source_file}")

    # Deduplicate exact paragraphs while preserving their order.
    paragraphs = []
    seen = set()

    for paragraph in (source_text + "\n\n" + EXTRA_TEXT).split("\n\n"):
        paragraph = normalize_paragraph(paragraph)

        if paragraph and paragraph.casefold() not in seen:
            seen.add(paragraph.casefold())
            paragraphs.append(paragraph)

    dataset = "\n\n".join(paragraphs) + "\n"
    OUTPUT_FILE.write_text(dataset, encoding="utf-8")

    print("\nMiniGPT v4 dataset created successfully.")
    print(f"Source: {source_file}")
    print(f"Unique paragraphs: {len(paragraphs):,}")
    print(f"Total characters: {len(dataset):,}")
    print(f"Unique characters: {len(set(dataset))}")
    print(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
