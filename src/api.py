
import json
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.model import MiniGPT


# ============================================================
# 1. PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

# Use the candidate temporarily for testing.
# Your original minigpt.pt remains untouched.
CHECKPOINT_PATH = BASE_DIR / "minigpt_candidate.pt"

RESULTS_DIR = BASE_DIR / "results"
HISTORY_PATH = RESULTS_DIR / "training_history_candidate.json"

DEVICE = torch.device("cpu")


# ============================================================
# 2. FASTAPI APP
# ============================================================

app = FastAPI(
    title="MiniGPT API",
    description="A small educational Transformer language model.",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"]
)


# ============================================================
# 3. LOAD CHECKPOINT
# ============================================================

def load_model():
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}"
        )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=False
    )

    required_keys = [
        "model_state_dict",
        "tokenizer_stoi",
        "tokenizer_itos",
        "vocab_size",
        "embed_size",
        "num_layers",
        "num_heads",
        "forward_expansion",
        "dropout",
        "max_length"
    ]

    missing = [
        key for key in required_keys
        if key not in checkpoint
    ]

    if missing:
        raise ValueError(
            f"Checkpoint is missing keys: {missing}"
        )

    model = MiniGPT(
        vocab_size=checkpoint["vocab_size"],
        embed_size=checkpoint["embed_size"],
        num_layers=checkpoint["num_layers"],
        num_heads=checkpoint["num_heads"],
        forward_expansion=checkpoint["forward_expansion"],
        dropout=checkpoint["dropout"],
        max_length=checkpoint["max_length"]
    ).to(DEVICE)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    stoi = checkpoint["tokenizer_stoi"]
    itos = checkpoint["tokenizer_itos"]

    if len(stoi) != checkpoint["vocab_size"]:
        raise ValueError("Tokenizer vocabulary size mismatch.")

    if len(itos) != checkpoint["vocab_size"]:
        raise ValueError("Tokenizer inverse vocabulary size mismatch.")

    return model, checkpoint, stoi, itos


model, checkpoint, stoi, itos = load_model()

print("MiniGPT checkpoint loaded:", CHECKPOINT_PATH.name)
print("Vocabulary size:", checkpoint["vocab_size"])
print("Model parameters:", sum(
    p.numel() for p in model.parameters()
))


# ============================================================
# 4. REQUEST VALIDATION
# ============================================================

class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=500)
    max_new_tokens: int = Field(default=100, ge=1, le=500)
    temperature: float = Field(default=0.6, gt=0, le=2.0)
    top_k: int = Field(default=5, ge=1, le=100)
    seed: int = Field(default=42, ge=0, le=2147483647)


# ============================================================
# 5. TEXT GENERATION
# ============================================================

@torch.inference_mode()
def generate_text(
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 0.6,
    top_k: int = 5,
    seed: int = 42
):
    unknown_chars = sorted(set(prompt) - set(stoi.keys()))

    if unknown_chars:
        raise ValueError(
            "Prompt contains characters outside the model vocabulary: "
            + repr(unknown_chars)
        )

    token_ids = [stoi[ch] for ch in prompt]

    if not token_ids:
        raise ValueError("Prompt contains no recognized characters.")

    x = torch.tensor(
        [token_ids],
        dtype=torch.long,
        device=DEVICE
    )

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    for _ in range(max_new_tokens):
        x_input = x[:, -checkpoint["max_length"]:]

        logits = model(x_input)
        logits = logits[:, -1, :] / temperature

        k = min(top_k, logits.shape[-1])
        top_values, _ = torch.topk(logits, k, dim=-1)
        cutoff = top_values[:, -1].unsqueeze(-1)

        logits = logits.masked_fill(
            logits < cutoff,
            float("-inf")
        )

        probabilities = torch.softmax(logits, dim=-1)

        next_token = torch.multinomial(
            probabilities,
            num_samples=1,
            generator=generator
        )

        x = torch.cat([x, next_token], dim=1)

    generated_ids = x[0].tolist()
    generated_text = "".join(itos[int(i)] for i in generated_ids)

    return {
        "generated_text": generated_text,
        "continuation": generated_text[len(prompt):]
    }


# ============================================================
# 6. HOME
# ============================================================

@app.get("/")
def home():
    return {
        "message": "MiniGPT API is running!",
        "model": "MiniGPT",
        "checkpoint": CHECKPOINT_PATH.name,
        "status": "ready",
        "docs": "/docs"
    }


# ============================================================
# 7. HEALTH CHECK
# ============================================================

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "checkpoint": CHECKPOINT_PATH.name,
        "device": str(DEVICE)
    }


# ============================================================
# 8. MODEL METRICS
# ============================================================

@app.get("/metrics")
def metrics():
    return {
        "model": "MiniGPT",
        "checkpoint": CHECKPOINT_PATH.name,
        "parameters": sum(
            p.numel() for p in model.parameters()
        ),
        "vocabulary_size": checkpoint["vocab_size"],
        "max_context_length": checkpoint["max_length"],
        "best_epoch": checkpoint.get("best_epoch"),
        "train_loss": checkpoint.get("train_loss"),
        "validation_loss": checkpoint.get("val_loss"),
        "train_perplexity": checkpoint.get("train_perplexity"),
        "validation_perplexity": checkpoint.get("val_perplexity")
    }


# ============================================================
# 9. TRAINING HISTORY
# ============================================================

@app.get("/training-history")
def training_history():
    if not HISTORY_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail="Candidate training history was not found."
        )

    try:
        with open(HISTORY_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not read training history: {exc}"
        )


# ============================================================
# 10. GENERATE ENDPOINT
# ============================================================

@app.post("/generate")
def generate(request: GenerateRequest):
    try:
        result = generate_text(
            prompt=request.prompt,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_k=request.top_k,
            seed=request.seed
        )

        return {
            "prompt": request.prompt,
            "generated_text": result["generated_text"],
            "continuation": result["continuation"],
            "settings": {
                "max_new_tokens": request.max_new_tokens,
                "temperature": request.temperature,
                "top_k": request.top_k,
                "seed": request.seed
            },
            "checkpoint": CHECKPOINT_PATH.name
        }

    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc)
        )


# ============================================================
# 11. OPTIONAL ABLATION RESULTS
# ============================================================

@app.get("/ablation")
def ablation():
    ablation_path = RESULTS_DIR / "ablation_results.json"

    if not ablation_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Ablation results file was not found."
        )

    try:
        with open(ablation_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not read ablation results: {exc}"
        )


# ============================================================
# 12. OPTIONAL RESULTS STATIC FILES
# ============================================================

if RESULTS_DIR.exists():
    app.mount(
        "/results",
        StaticFiles(directory=str(RESULTS_DIR)),
        name="results"
    )
