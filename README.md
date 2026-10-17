# MiniGPT: From-Scratch Transformer Language Model

## Overview

MiniGPT is a small character-level Transformer language model implemented
from scratch using PyTorch.

The project focuses on understanding the internal workings of Transformer
models rather than relying on high-level pretrained Transformer libraries.

## Architecture

Text
↓
Character Tokenizer
↓
Token Embeddings
↓
Positional Encoding
↓
Multi-Head Self-Attention
↓
Feed-Forward Network
↓
Layer Normalization
↓
Language Modeling Head
↓
Next-Token Prediction

## Features

- Character-level tokenization
- Token embeddings
- Positional encoding
- Scaled dot-product attention
- Multi-head attention
- Transformer blocks
- Residual connections
- Layer normalization
- Cross-entropy training
- AdamW optimization
- Validation evaluation
- Perplexity measurement
- Autoregressive text generation
- Model-size experiments
- Attention-head ablation study

## Experiments

### Model Capacity

We compared different Transformer configurations using:

- Training loss
- Validation loss
- Training perplexity
- Validation perplexity
- Parameter count
- Training time

### Attention Head Ablation

We tested:

- 1 attention head
- 2 attention heads
- 4 attention heads

The single-head configuration achieved the best validation perplexity
on our current small corpus.

## Key Finding

Increasing the number of attention heads reduced training loss but did not
improve validation performance on the small dataset.

This demonstrates why model architecture should be evaluated using
generalization metrics rather than training loss alone.

## Technologies

- Python
- PyTorch
- NumPy
- Matplotlib

## Project Structure

```text
MiniGPT/
├── data/
├── src/
│   ├── attention.py
│   ├── multi_head_attention.py
│   ├── positional_encoding.py
│   ├── tokenizer.py
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── generate.py
│   ├── experiment.py
│   └── ablation.py
├── minigpt.pt
└── README.md