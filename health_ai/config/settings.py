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
# Single-file GGUF. Update this when swapping models.
LLM_MODEL_PATH = MODEL_DIR / "Qwen3-4B-Q4_K_M.gguf"

# ── Speed settings ────────────────────────────────────────────────────────────

# KV cache size. 4096 gives ample room for RAG-augmented medical Q&A
# (system prompt ~200 tok + 3-4 chunks ~400 tok + user query + response).
LLM_CONTEXT_LENGTH = 4096

# Threads for token generation (decode phase)
# On CPU, logical hyperthreads compete for cache and floating-point units.
# Using physical cores (logical cores // 2) yields significantly better token-generation speed.
_logical_cores = _os.cpu_count() or 8
LLM_N_THREADS = max(1, _logical_cores // 2 if _logical_cores > 4 else _logical_cores)

# Threads for prompt processing (prefill phase) — can use more logical cores here
LLM_N_THREADS_BATCH = max(1, _logical_cores)

# Prompt eval batch — higher = faster prefill, uses more RAM
LLM_N_BATCH = 1024

# GPU layers. -1 = all on GPU. 0 = CPU only.
# Env-conditional so one settings.py works for both dev (GPU) and prod (CPU-only).
# Set HEALTH_AI_GPU_LAYERS=-1 on your dev machine; prod defaults to 0.
LLM_N_GPU_LAYERS = int(_os.environ.get("HEALTH_AI_GPU_LAYERS", "0"))

LLM_TEMPERATURE    = 0.2
LLM_TOP_P          = 0.9
LLM_REPEAT_PENALTY = 1.1    # slightly lower — less time spent on penalty calculation

# ── Token budgets — keep these LOW for speed ──────────────────────────────────
# Every token the model generates takes time. Cut off early.
# The model stops naturally at <|im_end|> before hitting the limit anyway.
MAX_TOKENS_GENERAL      = 2048
MAX_TOKENS_LAB          = 2048
MAX_TOKENS_PRESCRIPTION = 2048
MAX_TOKENS_SYMPTOM      = 2048
MAX_TOKENS_URGENT       = 512    # crisis responses should be short and directive
MAX_TOKENS_MENTAL_HEALTH = 2048

# ── Embedding model ───────────────────────────────────────────────────────────
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM        = 384

# ── Chunking ──────────────────────────────────────────────────────────────────
CHUNK_SIZE    = 500
CHUNK_OVERLAP = 100

# ── RAG ───────────────────────────────────────────────────────────────────────
TOP_K_CHUNKS = 4

# ── OCR ───────────────────────────────────────────────────────────────────────
OCR_MIN_CONFIDENCE = 0.40
OCR_MIN_LINE_CHARS = 2

# Stricter threshold for lab-report extraction where numeric accuracy matters.
# Values below this confidence are flagged as uncertain rather than trusted.
OCR_LAB_MIN_CONFIDENCE = 0.70
