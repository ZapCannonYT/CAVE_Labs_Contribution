"""
llm_loader.py — Qwen2.5-14B-Instruct GGUF singleton for Health AI v3.

Model: Qwen2.5-14B-Instruct-Q5_K_M (split GGUF, 3 shards)
Download: https://huggingface.co/Qwen/Qwen2.5-14B-Instruct-GGUF

Chat template (Qwen2.5 Instruct):
    <|im_start|>system\n{system}<|im_end|>
    <|im_start|>user\n{user}<|im_end|>
    <|im_start|>assistant\n
"""

import time
from threading import Lock

from health_ai.config.settings import (
    LLM_MODEL_PATH,
    LLM_CONTEXT_LENGTH,
    LLM_N_THREADS,
    LLM_N_THREADS_BATCH,
    LLM_N_BATCH,
    LLM_N_GPU_LAYERS,
    LLM_TEMPERATURE,
    LLM_TOP_P,
    LLM_REPEAT_PENALTY,
)
from health_ai.core.logger import get_logger
from health_ai.core.exceptions import ModelNotFoundError, ModelLoadError, GenerationError

log = get_logger(__name__)

_STOP_TOKENS = ["<|im_end|>", "<|endoftext|>", "<|im_start|>"]


def _build_prompt(system_prompt: str, user_prompt: str) -> str:
    return (
        f"<|im_start|>system\n{system_prompt.strip()}<|im_end|>\n"
        f"<|im_start|>user\n{user_prompt.strip()}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def _clean(text: str) -> str:
    for t in _STOP_TOKENS + ["</s>", "<|end|>"]:
        text = text.replace(t, "")
    return text.strip()


class LLMEngine:
    """
    Singleton wrapper for llama-cpp-python.
    Handles loading the model shards once and serialising access via lock.
    """

    _instance = None
    _lock     = Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._llm    = None
                    cls._instance._loaded = False
                    cls._instance._gen_lock = Lock()
        return cls._instance

    def _ensure_loaded(self):
        if self._loaded:
            return

        import os
        from llama_cpp import Llama

        if not LLM_MODEL_PATH.exists():
            raise ModelNotFoundError(
                f"Model file not found: {LLM_MODEL_PATH}\n"
                "Please ensure the GGUF model file is placed in the model/ directory."
            )

        model_name = LLM_MODEL_PATH.name
        if "00001-of-00003" in model_name:
            model_dir = LLM_MODEL_PATH.parent
            for shard in ["00002-of-00003", "00003-of-00003"]:
                p = model_dir / model_name.replace("00001-of-00003", shard)
                if not p.exists():
                    raise ModelNotFoundError(f"Missing model shard: {p.name}")


        log.info(f"Loading LLM: {LLM_MODEL_PATH.name} …")
        log.info(f"n_ctx={LLM_CONTEXT_LENGTH} n_threads={LLM_N_THREADS} "
                 f"n_threads_batch={LLM_N_THREADS_BATCH} n_batch={LLM_N_BATCH}")

        try:
            # We load the split GGUF by pointing to shard 1 (00001).
            # llama-cpp-python automatically detects and loads the other shards
            # if they share the same base name.
            self._llm = Llama(
                model_path=str(LLM_MODEL_PATH),
                n_ctx=LLM_CONTEXT_LENGTH,
                n_threads=LLM_N_THREADS,
                n_threads_batch=LLM_N_THREADS_BATCH,
                n_batch=LLM_N_BATCH,
                n_gpu_layers=LLM_N_GPU_LAYERS,
                verbose=False,
            )
            self._loaded = True
            log.info("LLM ready.")
        except Exception as e:
            raise ModelLoadError(f"Failed to load LLM: {e}") from e

    def generate(self, system_prompt: str, user_prompt: str, max_tokens: int = 300, stop_event=None, meta: dict = None) -> str:
        """
        Blocking generation. Returns the full response string.

        Always call this from a thread pool, not directly from an async handler:
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(None, lambda: llm.generate(...))
        """
        self._ensure_loaded()
        prompt = _build_prompt(system_prompt, user_prompt)

        try:
            with self._gen_lock:
                if meta is not None:
                    meta["t_start"] = time.time()
                # Use stream=True to support early cancellation via stop_event
                chunks = self._llm(
                    prompt,
                    max_tokens=max_tokens,
                    temperature=LLM_TEMPERATURE,
                    top_p=LLM_TOP_P,
                    repeat_penalty=LLM_REPEAT_PENALTY,
                    stop=_STOP_TOKENS,
                    echo=False,
                    stream=True,
                )
                text_parts = []
                for chunk in chunks:
                    if stop_event and stop_event.is_set():
                        log.info("LLM generation stopped early via stop_event.")
                        break
                    text_parts.append(chunk["choices"][0]["text"])
                return _clean("".join(text_parts))
        except Exception as e:
            raise GenerationError(f"Generation failed: {e}") from e
