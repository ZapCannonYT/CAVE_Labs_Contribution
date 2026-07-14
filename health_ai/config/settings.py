"""
settings.py — Central configuration for Health AI v3.
All tuneable values live here. No magic numbers scattered across the codebase.
"""
from pathlib import Path
import os as _os

# ── Directory layout ──────────────────────────────────────────────────────────

BASE_DIR  = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "model"
LOG_DIR   = BASE_DIR / "logs"

for _d in [MODEL_DIR, LOG_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ── Model path ────────────────────────────────────────────────────────────────
# Split GGUF — point to shard 1. llama-cpp loads the rest automatically.
# All 3 files must be in model/:
#   qwen2.5-14b-instruct-q5_k_m-00001-of-00003.gguf  ← point here
#   qwen2.5-14b-instruct-q5_k_m-00002-of-00003.gguf
#   qwen2.5-14b-instruct-q5_k_m-00003-of-00003.gguf
LLM_MODEL_PATH = MODEL_DIR / "qwen2.5-14b-instruct-q5_k_m-00001-of-00003.gguf"

# ── Speed settings ────────────────────────────────────────────────────────────

# KV cache size. 2048 is enough for medical Q&A.
# Every extra token here costs RAM and slows generation.
LLM_CONTEXT_LENGTH = 2048

# Threads for token generation (decode phase)
LLM_N_THREADS = max(4, (_os.cpu_count() or 8) - 2)

# Threads for prompt processing (prefill phase) — can use more cores here
LLM_N_THREADS_BATCH = max(4, (_os.cpu_count() or 8))

# Prompt eval batch — higher = faster prefill, uses more RAM
LLM_N_BATCH = 1024

# GPU layers. -1 = all on GPU. 0 = CPU only.
# Qwen2.5-14B Q5_K_M needs ~11GB VRAM for full offload.
# If you have 8GB VRAM, set to 20 (partial offload, still much faster than CPU).
LLM_N_GPU_LAYERS = -1

LLM_TEMPERATURE    = 0.2
LLM_TOP_P          = 0.9
LLM_REPEAT_PENALTY = 1.1    # slightly lower — less time spent on penalty calculation

# ── Token budgets — keep these LOW for speed ──────────────────────────────────
# Every token the model generates takes time. Cut off early.
# The model stops naturally at <|im_end|> before hitting the limit anyway.
MAX_TOKENS_GENERAL      = 250
MAX_TOKENS_LAB          = 400
MAX_TOKENS_PRESCRIPTION = 300
MAX_TOKENS_SYMPTOM      = 250

# ── Embedding model ───────────────────────────────────────────────────────────
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM        = 384

# ── Chunking ──────────────────────────────────────────────────────────────────
CHUNK_SIZE    = 500
CHUNK_OVERLAP = 100

# ── RAG ───────────────────────────────────────────────────────────────────────
TOP_K_CHUNKS = 5

# ── OCR ───────────────────────────────────────────────────────────────────────
OCR_MIN_CONFIDENCE = 0.40
OCR_MIN_LINE_CHARS = 2
