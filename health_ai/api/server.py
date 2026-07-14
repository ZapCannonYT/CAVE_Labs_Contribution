
"""
server.py — Health AI v3 FastAPI application.

Stateless architecture:
    - Phone stores all document chunks and embeddings
    - Server handles: OCR, embedding, and LLM generation
    - No profiles, no vector store on the server

Endpoints:
    POST /upload-and-embed      OCR + chunk + embed a file → return to client
    POST /embed-query           Embed a query string → return vector
    POST /generate              LLM generation from query + chunks
    GET  /health                Health check
    GET  /server/info           Server metadata
    GET  /greeting              Dr. Aria's welcome message
"""

import asyncio
import os
import socket
import tempfile
import re
import time as _time
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from health_ai.embeddings.embedder import EmbeddingModel
from health_ai.rag.chunker import TextChunker
from health_ai.utils.document_reader import DocumentReader, SUPPORTED_EXTENSIONS
from health_ai.model.llm_loader import LLMEngine
from health_ai.core.character import (
    classify_intent, get_system_prompt, get_max_tokens,
    detect_urgent, DISCLAIMER, URGENT_NOTICE, OFF_TOPIC_RESPONSE,
    GREETING_RESPONSE, MAX_HISTORY_TURNS, FAREWELL_RESPONSE,
)
from health_ai.core.safety import apply_safety_layer
from health_ai.rag.context_builder import build_context
from health_ai.core.logger import get_logger
from health_ai.core.exceptions import (
    ModelNotFoundError, UnsupportedFileTypeError,
    EmptyDocumentError, OCRError, EmbeddingError, GenerationError,
)
try:
    from health_ai.external.drug_api import get_drug_info
except Exception:
    def get_drug_info(name: str):
        return None

log = get_logger(__name__)

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Health AI v3",
    description="Offline personal medical AI — Dr. Aria powered by Qwen2.5-14B-Instruct.",
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Singletons ────────────────────────────────────────────────────────────────

_embedder: Optional[EmbeddingModel] = None
_llm:      Optional[LLMEngine]      = None
_chunker:  Optional[TextChunker]    = None
_reader:   Optional[DocumentReader] = None


def _get_local_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


@app.on_event("startup")
async def startup():
    global _embedder, _llm, _chunker, _reader

    port   = int(os.environ.get("PORT", 8000))
    lan_ip = _get_local_ip()

    print("\n" + "═" * 58)
    print("  🩺  Health AI v3  —  Dr. Aria is loading …")
    print("═" * 58)
    print(f"  Local:   http://localhost:{port}")
    print(f"  Network: http://{lan_ip}:{port}   ← use this in the app")
    print("═" * 58 + "\n")

    log.info("Loading embedding model …")
    _embedder = EmbeddingModel()
    _embedder._ensure_loaded()

    log.info("Loading LLM — this may take up to 60 seconds …")
    try:
        _llm = LLMEngine()
        _llm._ensure_loaded()
    except ModelNotFoundError as e:
        log.error(str(e))
        log.error("Server started but /generate will return 503 until model is placed.")

    _chunker = TextChunker()
    _reader  = DocumentReader()

    print("\n" + "═" * 58)
    print("  ✅  Dr. Aria is ready!")
    print("═" * 58 + "\n")


# ── Request / Response models ─────────────────────────────────────────────────

class EmbedQueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)


class EmbedQueryResponse(BaseModel):
    query:     str
    embedding: List[float]
    dim:       int


class MedicineItem(BaseModel):
    name:      Optional[str] = None
    dose:      Optional[str] = None
    type:      Optional[str] = None
    frequency: Optional[str] = None
    time:      Optional[str] = None
    meal:      Optional[str] = None
    taken:     Optional[int] = None
    reminder:  Optional[int] = None


class SymptomItem(BaseModel):
    name:       Optional[str] = None
    severity:   Optional[str] = None
    startedAt:  Optional[int] = None
    resolvedAt: Optional[int] = None
    notes:      Optional[str] = None


class PatientContext(BaseModel):
    medicines:       List[MedicineItem] = []
    activeSymptoms:  List[SymptomItem]  = []
    historySymptoms: List[SymptomItem]  = []


class GenerateRequest(BaseModel):
    query:           str                      = Field(..., min_length=1, max_length=2000)
    chunks:          List[str]                = Field(default=[])
    history:         List[str]                = Field(default=[])
    patient_context: Optional[PatientContext] = None


class GenerateResponse(BaseModel):
    response: str
    intent:   str


class ChunkOut(BaseModel):
    text:      str
    embedding: List[float]
    metadata:  dict


class UploadResponse(BaseModel):
    status:      str
    filename:    str
    doc_type:    str
    chunk_count: int
    chunks:      List[ChunkOut]


# ══════════════════════════════════════════════════════════════════════════════
# LOCAL-DATA HELPERS — medicine and symptom query handling
# ══════════════════════════════════════════════════════════════════════════════

# ── Keyword sets ──────────────────────────────────────────────────────────────

_MED_KW = frozenset([
    "medicine", "medicines", "medication", "medications",
    "meds", "med", "pill", "pills",
    "tablet", "tablets", "capsule", "capsules",
    "drug", "drugs", "prescription", "prescriptions",
    "syrup", "syrups", "injection", "injections",
    "supplement", "supplements", "vitamin", "vitamins",
    "antibiotic", "antibiotics", "steroid", "steroids", "insulin",
    "what am i taking", "what am i on", "what do i take",
    "what are my", "list my", "show my", "tell me my",
    "explain my", "about my meds", "about my medication",
    "my treatment", "my therapy", "my drugs", "my pills",
    "am i taking", "am i on any", "do i take",
    "side effect", "side effects", "adverse", "interaction",
    "interactions", "drug interaction", "contraindication",
    "dosage", "dose", "overdose",
    "how much should i take", "when should i take",
    "should i take with food",
    "morning medicine", "evening medicine", "night medicine",
    "before food", "after food", "with food", "on empty",
    "how many medicines", "how many medications", "how many pills",
    "what medicines", "which medicines", "which medication",
])

_SYM_KW = frozenset([
    "symptom", "symptoms", "complaint", "complaints",
    "condition", "conditions", "issue", "issues",
    "problem", "problems", "illness", "sickness",
    "ailment", "ailments", "discomfort",
    "how am i feeling", "how am i doing", "how do i feel",
    "what is wrong", "what's wrong", "whats wrong",
    "not feeling well", "feeling sick", "feeling unwell",
    "active symptoms", "current symptoms", "my symptoms",
    "what do i have", "health issues", "my health issues",
    "health problems", "my complaints",
    "analyse my", "analyze my", "look at my symptoms",
    "check my symptoms", "about my symptoms",
    "regarding my symptoms", "review my symptoms",
    "how serious", "how bad", "how severe",
    "should i be worried", "is it serious", "is it dangerous",
    "how long have i", "how long has",
    "what can i do", "what should i do",
    "home remedy", "home remedies", "relief",
])

# Food / lifestyle keywords that trigger cross-domain reasoning
_FOOD_LIFESTYLE_KW = frozenset([
    "eat", "eating", "food", "foods", "drink", "drinking",
    "alcohol", "coffee", "tea", "dairy", "milk", "ice cream",
    "juice", "fruit", "vegetable", "diet", "meal", "snack",
    "exercise", "workout", "gym", "run", "running", "sport",
    "sleep", "rest", "fast", "fasting", "smoke", "smoking",
    "can i have", "can i eat", "can i drink", "is it safe to eat",
    "is it ok to", "safe to", "avoid", "avoid while",
    "while taking", "while on", "during treatment",
    "what to avoid", "foods to avoid", "what not to eat",
    "what should i avoid", "lifestyle", "habits",
])

# Cross-domain: queries that explicitly reference BOTH medicines AND symptoms
_CROSS_DOMAIN_KW = frozenset([
    "causing", "cause", "caused by", "side effect of",
    "because of my", "due to my", "related to my",
    "from my medication", "from my medicine", "from taking",
    "medicine causing", "medication causing", "pills causing",
    "meds causing", "is it my medicine", "is it my medication",
    "could it be my", "might it be my",
    "symptom from", "reaction to",
    "together", "combination", "combining",
    "affect my symptoms", "affect my medicines",
    "interact with my symptoms",
])


# ── Detection functions ───────────────────────────────────────────────────────

def _is_medicine_query(q: str) -> bool:
    q = q.lower().strip()

    # 🔥 Strong direct signals (handles real user language)
    if any(x in q for x in [
        "meds", "medicine", "medication",
        "pill", "tablet", "capsule",
        "what am i taking", "what am i on",
        "what meds", "my meds", "my medicine",
        "what do i take", "what meds am i on"
    ]):
        return True

    # 🧠 Fallback to full keyword set
    return any(kw in q for kw in _MED_KW)


def _is_symptom_query(q: str) -> bool:
    return any(kw in q for kw in _SYM_KW)


def _is_food_lifestyle_query(q: str) -> bool:
    return any(kw in q for kw in _FOOD_LIFESTYLE_KW)


def _is_cross_domain_query(q: str) -> bool:
    """True when query explicitly links medicines with symptoms."""
    has_med = _is_medicine_query(q)
    has_sym = _is_symptom_query(q)
    has_cross = any(kw in q for kw in _CROSS_DOMAIN_KW)
    return has_cross and (has_med or has_sym)


# ── Item lookup helpers ───────────────────────────────────────────────────────

def _find_mentioned_medicine(q: str, medicines):
    return [m for m in medicines if (m.name or "").strip().lower() in q]


def _find_mentioned_symptom(q: str, symptoms):
    return [s for s in symptoms if (s.name or "").strip().lower() in q]


# ── Formatting helpers ────────────────────────────────────────────────────────

def _duration_label(started_at_ms) -> str:
    if not started_at_ms:
        return ""
    try:
        diff_ms   = int(_time.time() * 1000) - int(started_at_ms)
        diff_mins = diff_ms // 60000
        if diff_mins < 60:    return f"{diff_mins}m"
        diff_hrs = diff_mins  // 60
        if diff_hrs < 24:     return f"{diff_hrs}h"
        diff_days = diff_hrs  // 24
        if diff_days < 7:     return f"{diff_days}d"
        return f"{diff_days // 7}w"
    except Exception:
        return ""


def _severity_order(s) -> int:
    return {"critical": 0, "severe": 0, "high": 1,
            "moderate": 2, "medium": 2, "mild": 3, "low": 4}.get(
        (s.severity or "").lower(), 5)


def _build_medicine_context(medicines) -> str:
    lines = []
    for m in medicines:
        if not (m.name or "").strip():
            continue
        row     = f"• {m.name.strip()}"
        details = []
        if m.dose:      details.append(f"dose: {m.dose}")
        if m.type:      details.append(f"type: {m.type}")
        if m.frequency: details.append(f"frequency: {m.frequency}")
        if m.time:      details.append(f"time: {m.time}")
        if m.meal:      details.append(f"meal: {m.meal}")
        if details:
            row += "\n  " + " | ".join(details)
        lines.append(row)
    return "\n".join(lines) if lines else ""


def _build_enriched_medicine_context(medicines) -> str:
    """Build medicine context with optional external drug info enrichment."""
    lines = []
    for m in medicines:
        if not (m.name or "").strip():
            continue
        try:
            info = get_drug_info(m.name or "")
        except Exception:
            info = None
        row = f"• {m.name.strip()}"
        if m.dose:
            row += f" ({m.dose})"
        details = []
        if info and isinstance(info, dict) and info.get("rxcui"):
            details.append("verified drug info available")
        if m.frequency:
            details.append(f"frequency: {m.frequency}")
        if m.time:
            details.append(f"time: {m.time}")
        if m.meal:
            details.append(f"meal: {m.meal}")
        if details:
            row += "\n  " + " | ".join(details)
        lines.append(row)
    return "\n".join(lines) if lines else _build_medicine_context(medicines)


def _build_symptom_context(symptoms) -> str:
    lines = []
    for s in sorted(symptoms, key=_severity_order):
        if not (s.name or "").strip():
            continue
        row  = f"• {s.name.strip()}"
        meta = []
        if s.severity: meta.append(f"severity: {s.severity}")
        dur = _duration_label(s.startedAt)
        if dur:        meta.append(f"duration: {dur}")
        if s.notes:    meta.append(f"notes: {s.notes}")
        if meta:
            row += "\n  " + " | ".join(meta)
        lines.append(row)
    return "\n".join(lines) if lines else ""


# ── LLM call helper ───────────────────────────────────────────────────────────

async def _ask_llm(prompt: str, max_tokens: int = 220) -> str:
    """Run a blocking LLM call in the thread pool. Returns the response string."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None, lambda: _llm.generate(prompt, "", max_tokens=max_tokens)
    )


# ─────────────────────────────────────────────────────────────────────────────
# QUERY HANDLERS
# ─────────────────────────────────────────────────────────────────────────────

# ── 1. Medicine queries ───────────────────────────────────────────────────────



async def _handle_medicine_query(query: str, q: str, medicines) -> str:
    valid_meds = [m for m in medicines if (m.name or "").strip()]

    if not valid_meds:
        return (
            "I don't see any medicines in your records yet. 💊\n\n"
            "You can add them from the **Medication Vault** on the home screen "
            "and I'll help you understand them right away!"
        )

    enriched_data = []
    for m in valid_meds:
        info = None
        try:
            info = get_drug_info(m.name or "")
        except Exception:
            info = None
        enriched_data.append({
            "name": m.name,
            "dose": m.dose,
            "time": m.time,
            "meal": m.meal,
            "api_info": info,
        })

    def _clean_text(text: str) -> str:
        text = (text or "").replace("###", " ").replace("**", " ").replace("_", " ")
        text = text.replace("\r", " ").replace("\n", " ")
        text = text.replace("- ", " ").replace("•", " ")
        text = " ".join(text.split())
        return text.strip(" .;:")

    def _first_sentence(text: str) -> str:
        clean = _clean_text(text)
        if not clean:
            return ""
        parts = re.split(r'(?<=[.!?])\s+', clean)
        return parts[0].strip() if parts else clean

    async def _explain_medicine(name: str, dose: Optional[str], api_info: Optional[dict]) -> str:
        drug_hint = ""
        if api_info and isinstance(api_info, dict):
            if api_info.get("rxcui"):
                drug_hint = "Verified drug metadata is available."
            elif api_info.get("label"):
                drug_hint = "Verified label information is available."
        prompt = f"""You are Dr. Aria, a warm and simple health assistant.

Medicine: {name}
Dose: {dose or "not listed"}
{drug_hint}

Give exactly 2 short sentences:
1. What it does
2. Why it is used

No bullets, no headings, no markdown, no extra commentary.
Keep it friendly and easy to understand."""
        try:
            raw = await _ask_llm(prompt, max_tokens=80)
        except Exception:
            raw = ""

        sentence = _first_sentence(raw)
        if not sentence:
            sentence = "Helps manage the condition it was prescribed for."
        return sentence

    def _format_when_to_take(time_value: Optional[str], meal_value: Optional[str]) -> str:
        if time_value and meal_value:
            return f"{time_value} ({meal_value})"
        if time_value:
            return time_value
        if meal_value:
            return meal_value
        return "as prescribed"

    specific = _find_mentioned_medicine(q, valid_meds)
    target_meds = specific if specific else valid_meds

    if specific:
        blocks = []
        for i, med in enumerate(target_meds, start=1):
            api_info = None
            for item in enriched_data:
                if (item["name"] or "").strip().lower() == (med.name or "").strip().lower():
                    api_info = item.get("api_info")
                    break

            purpose = await _explain_medicine(med.name or "Unknown medicine", med.dose, api_info)
            when = _format_when_to_take(med.time, med.meal)
            dose = f" ({med.dose})" if med.dose else ""

            blocks.append(
                f"{i}. 💊 **{med.name}{dose}**\n"
                f"   • Purpose: {purpose}.\n"
                f"   • When to take: {when}."
            )

        return "\n\n".join(blocks) + "\n\nYou're doing a great job keeping track of your medicines 👍"

    if any(kw in q for kw in ("how many", "count", "number of")):
        count = len(valid_meds)
        lines = "\n".join(f"{i}. 💊 **{m.name}**" + (f" ({m.dose})" if m.dose else "")
                          for i, m in enumerate(valid_meds, start=1))
        return (
            f"You have **{count} medicine{'s' if count != 1 else ''}** logged 💊\n\n"
            + lines
            + "\n\nAsk me about any of them and I'll explain what they do!"
        )

    time_filter = None
    if any(kw in q for kw in ("morning", "breakfast")):
        time_filter = "morning"
    elif any(kw in q for kw in ("evening", "dinner", "night", "bedtime")):
        time_filter = "evening"
    elif any(kw in q for kw in ("afternoon", "lunch")):
        time_filter = "afternoon"

    if time_filter:
        filtered = [m for m in valid_meds if time_filter in (m.time or "").lower()]
        if filtered:
            lines = []
            for i, med in enumerate(filtered, start=1):
                dose = f" ({med.dose})" if med.dose else ""
                lines.append(f"{i}. 💊 **{med.name}{dose}**")
            return (
                f"Your **{time_filter}** medicines ⏰\n\n"
                + "\n\n".join(lines)
                + "\n\nYou can ask me about any medicine here for a short explanation."
            )
        return (
            f"I don't see any medicines specifically scheduled for the {time_filter} "
            f"in your records. You have {len(valid_meds)} medicine(s) logged — "
            "check the **Medication Vault** for the full schedule. 📋"
        )

    if any(kw in q for kw in ("side effect", "side effects", "adverse",
                               "interaction", "interactions", "contraindication")):
        blocks = []
        for i, med in enumerate(valid_meds, start=1):
            prompt = f"""You are Dr. Aria, a warm and helpful health assistant.

Medicine: {med.name}
Dose: {med.dose or "not listed"}

Give exactly 2 common side effects in simple language.
No bullets, no headings, no markdown, no extra commentary.
Keep it short and calm."""
            try:
                raw = await _ask_llm(prompt, max_tokens=60)
            except Exception:
                raw = ""

            clean = _clean_text(raw)
            if not clean:
                clean = "Mild stomach upset or headache"

            blocks.append(
                f"{i}. 💊 **{med.name}**" + (f" ({med.dose})" if med.dose else "") + "\n"
                f"   • Common side effects: {clean}."
            )

        return "\n\n".join(blocks) + "\n\nAlways check with your doctor before making changes 😊"

    blocks = []
    for i, med in enumerate(valid_meds, start=1):
        api_info = None
        for item in enriched_data:
            if (item["name"] or "").strip().lower() == (med.name or "").strip().lower():
                api_info = item.get("api_info")
                break

        purpose = await _explain_medicine(med.name or "Unknown medicine", med.dose, api_info)
        when = _format_when_to_take(med.time, med.meal)
        dose = f" ({med.dose})" if med.dose else ""

        blocks.append(
            f"{i}. 💊 **{med.name}{dose}**\n"
            f"   • Purpose: {purpose}.\n"
            f"   • When to take: {when}."
        )

    return "\n\n".join(blocks) + "\n\nYou're doing a great job keeping track of your medicines 👍"




async def _handle_symptom_query(query: str, q: str,
                                 active_symptoms, history_symptoms) -> str:
    valid_active  = [s for s in active_symptoms  if (s.name or "").strip()]
    valid_history = [s for s in history_symptoms if (s.name or "").strip()]

    if not valid_active and not valid_history:
        return (
            "I don't see any symptoms logged yet. 🩺\n\n"
            "You can add them using the **Symptom Log** on the home screen. "
            "Once logged, I can give you helpful insights!"
        )

    # ── History / resolved symptoms ───────────────────────────────────────────
    if any(kw in q for kw in ("history", "past", "previous", "resolved",
                               "old symptoms", "before", "had before")):
        if not valid_history:
            return "You don't have any resolved symptoms in your history yet. ✅"
        ctx = _build_symptom_context(valid_history)
        return await _ask_llm(f"""You are Dr. Aria, a warm and supportive health assistant. 🩺

The user is reviewing their past symptoms.

Resolved symptom history:
{ctx}

User question: {query}

Rules:
- Give a brief, friendly summary
- Point out any patterns if obvious (e.g., recurring headaches)
- MAX 3 bullet points, MAX 100 words
- Warm, reassuring tone""", max_tokens=180)

    # ── Specific symptom named ────────────────────────────────────────────────
    specific = _find_mentioned_symptom(q, valid_active)
    if not specific:
        specific = _find_mentioned_symptom(q, valid_history)
    if specific:
        ctx = _build_symptom_context(specific)
        return await _ask_llm(f"""You are Dr. Aria, a warm and supportive health assistant. 🩺

The user is asking about a specific symptom.

Symptom record:
{ctx}

User question: {query}

Reply in this format:

🩺 **[Symptom name]**
• **Likely cause:** One short line.
• **What you can do:** One practical home-care tip.
• **See a doctor if:** One clear warning sign.

Rules:
- MAX 100 words
- Flag high/critical severity with ⚠️
- Calm and reassuring — not alarming""", max_tokens=200)

    # ── No active symptoms ────────────────────────────────────────────────────
    if not valid_active:
        return (
            "Great news — you have no active symptoms right now! ✅\n\n"
            + (f"You have **{len(valid_history)}** resolved symptom(s) in your history. "
               "Ask me about them anytime!" if valid_history else "")
        )

    # ── Count / list ──────────────────────────────────────────────────────────
    if any(kw in q for kw in ("how many", "list", "count", "show")):
        count = len(valid_active)
        lines = "\n".join(
            f"• **{s.name}**"
            + (f" ({s.severity})" if s.severity else "")
            + (f" — {_duration_label(s.startedAt)}" if s.startedAt else "")
            for s in sorted(valid_active, key=_severity_order)
        )
        return f"You have **{count} active symptom{'s' if count != 1 else ''}** 🩺\n\n{lines}"

    # ── Severity / seriousness ────────────────────────────────────────────────
    if any(kw in q for kw in ("serious", "severe", "bad", "dangerous",
                               "worried", "worry", "concern", "worse")):
        critical = [s for s in valid_active
                    if (s.severity or "").lower() in ("critical", "severe", "high")]
        if critical:
            ctx = _build_symptom_context(critical)
            return await _ask_llm(f"""You are Dr. Aria, a warm but precise health assistant. 🩺

The user is concerned about the severity of their symptoms.

High-severity symptoms:
{ctx}

Rules:
- Be honest but calm — not alarming
- For each: state the concern level and ONE clear action
- Recommend seeing a doctor where appropriate
- MAX 3 bullet points, MAX 100 words
- Use ⚠️ for high, 🔴 for critical""", max_tokens=180)
        return (
            "Your current symptoms are not flagged as high severity. ✅\n\n"
            "That said, trust how you feel — if anything gets worse, "
            "don't hesitate to see a doctor. Take care! 😊"
        )

    # ── Duration ──────────────────────────────────────────────────────────────
    if any(kw in q for kw in ("how long", "since when", "duration",
                               "started", "days ago", "began")):
        lines = []
        for s in sorted(valid_active, key=_severity_order):
            dur  = _duration_label(s.startedAt)
            line = f"• **{s.name}**"
            if dur: line += f" — for **{dur}**"
            if s.severity: line += f" ({s.severity})"
            lines.append(line)
        return "Here's how long your symptoms have been active ⏱️\n\n" + "\n".join(lines)

    # ── Home remedies / what to do ────────────────────────────────────────────
    if any(kw in q for kw in ("what can i do", "what should i do",
                               "home remedy", "home remedies", "relief",
                               "treat", "treatment", "help with")):
        ctx = _build_symptom_context(valid_active)
        return await _ask_llm(f"""You are Dr. Aria, a warm and practical health assistant. 🩺

The user wants home-care advice for their symptoms.

Active symptoms:
{ctx}

Reply in this format:

For each symptom:
🌿 **[Symptom]:** One specific, practical home-care tip.

⚠️ **See a doctor if:**
• [Warning sign 1]
• [Warning sign 2]

Rules:
- MAX 5 items total, MAX 120 words
- Practical and specific — not generic
- Warm, helpful tone
- End with: "See a doctor if symptoms worsen or persist beyond 48 hours. 💙" """, max_tokens=200)

    # ── Default: full symptom overview ────────────────────────────────────────
    ctx = _build_symptom_context(valid_active)
    has_critical = any((s.severity or "").lower() in ("critical", "severe")
                       for s in valid_active)
    severity_note = (
        "⚠️ One or more symptoms are flagged critical/severe. Recommend seeing a doctor."
        if has_critical else ""
    )

    return await _ask_llm(f"""You are Dr. Aria, a warm and supportive health assistant. 🩺

Active symptoms:
{ctx}
{severity_note}

User question: {query}

Reply in this exact format:

📋 **Here's a summary of your symptoms:**

For each symptom:
• **[Name]** [severity icon: ✅mild / ⚠️moderate / 🔴critical]
  — Likely cause: [1 short line]
  — What to do: [1 actionable tip]

⚠️ **See a doctor if:** [2 clear warning signs]

Rules:
- MAX 130 words
- Short, clear sentences
- Friendly and reassuring — not clinical
- Flag critical symptoms with 🔴""", max_tokens=260)


# ── 3. Food / lifestyle interaction queries ───────────────────────────────────


async def _handle_food_lifestyle_query(query: str, q: str, medicines, active_symptoms) -> str:
    """
    Handle queries like:
    - "can I eat ice cream while taking metformin?"
    - "can I drink alcohol on antibiotics?"
    - "is it safe to exercise with my current medicines?"
    - "what foods should I avoid?"
    """
    valid_meds = [m for m in medicines if (m.name or "").strip()]
    valid_symptoms = [s for s in active_symptoms if (s.name or "").strip()]

    if not valid_meds and not valid_symptoms:
        return (
            "I don't see any medicines or symptoms in your records to check against. 💊🩺\n\n"
            "Add them from the **Medication Vault** and **Symptom Log** and I can give you "
            "personalised food and lifestyle advice!"
        )

    med_ctx = _build_enriched_medicine_context(valid_meds) if valid_meds else "None"
    sym_ctx = _build_symptom_context(valid_symptoms) if valid_symptoms else "None"

    # Check if a specific medicine was mentioned
    specific = _find_mentioned_medicine(q, valid_meds)
    if specific:
        med_ctx = _build_enriched_medicine_context(specific)

    return await _ask_llm(f"""You are Dr. Aria, a warm and knowledgeable health assistant. 🩺

The user has a food or lifestyle question related to their medicines and symptoms.

Their medicines:
{med_ctx}

Their symptoms:
{sym_ctx}

User question: {query}

Reply in this format:

🍽️ **Quick answer:** [Yes / No / It depends — one sentence]

• **Why:** [One clear explanation based on their specific medicines and symptoms]
• **What to do:** [One practical, specific tip]
• **Good to know:** [One helpful bonus tip, if relevant]

Rules:
- Base your answer on the medicines and symptoms listed above
- If the medicine isn't known to interact with the food/activity, say so clearly
- If there are no relevant symptoms, keep the answer focused on medicines only
- MAX 100 words
- Friendly, reassuring tone — not scary
- End with: "When in doubt, always check with your doctor or pharmacist! 😊" """, max_tokens=200)

# ── 4. Cross-domain: medicine + symptom reasoning ────────────────────────────


async def _handle_cross_domain_query(query: str, q: str,
                                      medicines, active_symptoms) -> str:
    """
    Handle queries that link medicines and symptoms together, such as:
    - "could my metformin be causing my nausea?"
    - "are my symptoms a side effect of my antibiotics?"
    - "I started a new medicine and now I feel dizzy — is it related?"
    - "my headache started after I began taking this tablet"
    """
    valid_meds = [m for m in medicines if (m.name or "").strip()]
    valid_syms = [s for s in active_symptoms if (s.name or "").strip()]

    # If we only have one side, route to the right handler
    if not valid_meds and not valid_syms:
        return (
            "I'd love to help connect the dots, but I don't see any medicines "
            "or symptoms in your records yet. 🩺\n\n"
            "Add them from the home screen and I can give you personalised insights!"
        )
    if not valid_meds:
        return await _handle_symptom_query(query, q, active_symptoms, [])
    if not valid_syms:
        return await _handle_medicine_query(query, q, medicines)

    med_ctx = _build_medicine_context(valid_meds)
    sym_ctx = _build_symptom_context(valid_syms)

    # Check for specific items mentioned
    specific_med = _find_mentioned_medicine(q, valid_meds)
    specific_sym = _find_mentioned_symptom(q, valid_syms)
    focused_med_ctx = _build_medicine_context(specific_med) if specific_med else med_ctx
    focused_sym_ctx = _build_symptom_context(specific_sym) if specific_sym else sym_ctx

    return await _ask_llm(f"""You are Dr. Aria, a warm and analytical health assistant. 🩺

The user wants to understand whether their medicines and symptoms might be connected.

Their medicines:
{focused_med_ctx}

Their active symptoms:
{focused_sym_ctx}

User question: {query}

Think carefully and reply in this format:

🔍 **Connection Analysis**

• **Likely related?** [Yes / Possibly / Unlikely — one sentence explaining why]
• **What this means:** [One clear, reassuring explanation]
• **What to do:** [One concrete next step — e.g., "mention this to your doctor at your next visit"]

⚕️ **Important:** [One brief safety note if needed, otherwise skip this line]

Rules:
- Reason specifically about the medicines and symptoms listed
- If a medicine is known to cause a listed symptom as a side effect, say so clearly
- If the connection is unlikely, reassure the user
- MAX 120 words
- Calm, analytical, and friendly — not alarming
- Never diagnose — only reason about possible connections
- End with: "Your doctor is the best person to confirm this. 💙" """, max_tokens=240)


# ══════════════════════════════════════════════════════════════════════════════
# ROUTING LOGIC — decides which handler to call
# ══════════════════════════════════════════════════════════════════════════════


async def _route_patient_context_query(
    query: str, q: str,
    medicines, active_symptoms, history_symptoms
) -> Optional[GenerateResponse]:
    """
    Inspect the query and patient context, route to the right handler.
    Returns a GenerateResponse if handled locally, or None to fall through
    to the normal LLM path.
    """

    has_meds = bool([m for m in medicines if (m.name or "").strip()])
    has_syms = bool([s for s in active_symptoms if (s.name or "").strip()])

    # ── Food / lifestyle interaction ──────────────────────────────────────────
    # These queries should be handled before cross-domain reasoning so that
    # "can I eat ice cream?" can use both the symptom and medicine context.
    if _is_food_lifestyle_query(q) and (has_meds or has_syms):
        log.info(f"Food/lifestyle query: {query!r}")
        response = await _handle_food_lifestyle_query(query, q, medicines, active_symptoms)
        return GenerateResponse(response=response, intent="lifestyle_interaction")

    # ── Cross-domain: medicine + symptom connection ───────────────────────────
    if _is_cross_domain_query(q) and (has_meds or has_syms):
        log.info(f"Cross-domain query: {query!r}")
        response = await _handle_cross_domain_query(
            query, q, medicines, active_symptoms)
        return GenerateResponse(response=response, intent="cross_domain")

    # ── Pure medicine query ───────────────────────────────────────────────────
    if _is_medicine_query(q):
        log.info(f"Medicine query: {query!r}")
        response = await _handle_medicine_query(query, q, medicines)
        return GenerateResponse(response=response, intent="medication_info")

    # ── Pure symptom query ────────────────────────────────────────────────────
    if _is_symptom_query(q):
        log.info(f"Symptom query: {query!r}")
        response = await _handle_symptom_query(
            query, q, active_symptoms, history_symptoms)
        return GenerateResponse(response=response, intent="symptom_info")

    return None  # fall through to normal LLM



# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/greeting", tags=["System"])
async def greeting():
    """Returns Dr. Aria's introduction message. Call on app startup."""
    return {"message": GREETING_RESPONSE, "character": "Dr. Aria"}


@app.get("/health", tags=["System"])
async def health():
    return {
        "status":          "ok",
        "version":         "3.0.0",
        "llm_loaded":      (_llm is not None and _llm._loaded),
        "embedder_loaded": (_embedder is not None and _embedder._loaded),
        "model":           "Qwen2.5-14B-Instruct-Q5_K_M",
    }


@app.get("/server/info", tags=["System"])
async def server_info():
    port = int(os.environ.get("PORT", 8000))
    return {
        "server":          "Health AI v3",
        "character":       "Dr. Aria",
        "lan_ip":          _get_local_ip(),
        "port":            port,
        "llm_ready":       (_llm is not None and _llm._loaded),
        "embedder_ready":  (_embedder is not None and _embedder._loaded),
        "model":           "Qwen2.5-14B-Instruct-Q5_K_M",
        "embedding_model": "all-MiniLM-L6-v2",
    }


@app.post("/upload-and-embed", response_model=UploadResponse, tags=["Documents"])
async def upload_and_embed(file: UploadFile = File(...)):
    filename = file.filename or "upload"
    ext      = Path(filename).suffix.lower()

    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{ext}'. "
                   f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}",
        )

    suffix = ext or ".tmp"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        log.info(f"Processing upload: {filename}")
        try:
            text = _reader.extract(tmp_path)
        except UnsupportedFileTypeError as e:
            raise HTTPException(status_code=415, detail=str(e))
        except EmptyDocumentError as e:
            raise HTTPException(status_code=422, detail=str(e))
        except OCRError as e:
            raise HTTPException(status_code=422, detail=str(e))

        doc_type = _reader.detect_doc_type(filename)
        chunks   = _chunker.chunk(text, {"filename": filename, "doc_type": doc_type})

        if not chunks:
            raise HTTPException(status_code=422,
                                detail="Document produced no text chunks.")

        try:
            embeddings = _embedder.embed([c.text for c in chunks])
        except EmbeddingError as e:
            raise HTTPException(status_code=500, detail=f"Embedding failed: {e}")

        log.info(f"✅ {filename} → {len(chunks)} chunks, doc_type={doc_type}")

        return UploadResponse(
            status="success",
            filename=filename,
            doc_type=doc_type,
            chunk_count=len(chunks),
            chunks=[
                ChunkOut(
                    text=c.text,
                    embedding=embeddings[i].tolist(),
                    metadata=c.metadata,
                )
                for i, c in enumerate(chunks)
            ],
        )
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


@app.post("/embed-query", response_model=EmbedQueryResponse, tags=["RAG"])
async def embed_query(request: EmbedQueryRequest):
    try:
        vec = _embedder.embed_single(request.query)
    except EmbeddingError as e:
        raise HTTPException(status_code=500, detail=str(e))
    return EmbedQueryResponse(query=request.query, embedding=vec, dim=len(vec))



@app.post("/generate", response_model=GenerateResponse, tags=["Generation"])
async def generate(request: GenerateRequest):

    if _llm is None or not _llm._loaded:
        raise HTTPException(status_code=503,
                            detail="LLM not loaded. Check /health for status.")

    # ── Instant intent responses — no LLM needed ──────────────────────────────
    intent = classify_intent(request.query)

    # greeting/farewell still first (safe)
    if intent == "greeting":
        return GenerateResponse(response=GREETING_RESPONSE, intent="greeting")

    if intent == "farewell":
        return GenerateResponse(response=FAREWELL_RESPONSE, intent="farewell")

    # ── Extract patient context ───────────────────────────────────────────────
    medicines = []
    active_symptoms = []
    history_symptoms = []

    if request.patient_context:
        medicines = request.patient_context.medicines or []
        active_symptoms = request.patient_context.activeSymptoms or []
        history_symptoms = getattr(request.patient_context, "historySymptoms", []) or []

    q = request.query.lower()

    # ── Patient-context routing ───────────────────────────────────────────────
    # Only engaged when patient_context is provided.
    # Returns a response if the query is about medicines, symptoms, food/lifestyle,
    # or cross-domain reasoning. Otherwise falls through to the normal LLM path.
    if request.patient_context:
        local_response = await _route_patient_context_query(
            request.query, q, medicines, active_symptoms, history_symptoms
        )
        if local_response is not None:
            return local_response

    # ── Off-topic only after patient-context checks ───────────────────────────
    if intent == "off_topic":
        log.info(f"Off-topic rejected: {request.query!r}")
        return GenerateResponse(response=OFF_TOPIC_RESPONSE, intent="off_topic")

    # ── Normal LLM path ───────────────────────────────────────────────────────
    trimmed_history = (request.history[-(MAX_HISTORY_TURNS * 2):]
                       if request.history else [])

    system_prompt = get_system_prompt(intent)
    max_tokens    = get_max_tokens(intent)
    user_prompt   = build_context(request.query, request.chunks, trimmed_history)

    log.info(f"Generate [{intent}] — {len(request.chunks)} chunks, "
             f"history={len(trimmed_history)//2} turns, max_tokens={max_tokens}")

    loop = asyncio.get_running_loop()
    try:
        response = await loop.run_in_executor(
            None,
            lambda: _llm.generate(system_prompt, user_prompt, max_tokens=max_tokens),
        )
    except GenerationError as e:
        raise HTTPException(status_code=500, detail=str(e))

    response = apply_safety_layer(response, request.query)
    return GenerateResponse(response=response, intent=intent)


# ── v2 compatibility routes ───────────────────────────────────────────────────


@app.post("/generate/{profile_id}", response_model=GenerateResponse, tags=["v2 compat"])
async def generate_compat(profile_id: str, request: GenerateRequest):
    log.info(f"v2 compat: /generate/{profile_id}")
    return await generate(request)


@app.post("/query/{profile_id}", response_model=GenerateResponse, tags=["v2 compat"])
async def query_compat(profile_id: str, request: GenerateRequest):
    log.info(f"v2 compat: /query/{profile_id}")
    return await generate(request)


@app.post("/upload-and-embed/{profile_id}", response_model=UploadResponse,
          tags=["v2 compat"])
async def upload_compat(profile_id: str, file: UploadFile = File(...)):
    log.info(f"v2 compat: /upload-and-embed/{profile_id}")
    return await upload_and_embed(file)


@app.post("/embed-query/{profile_id}", response_model=EmbedQueryResponse,
          tags=["v2 compat"])
async def embed_query_compat(profile_id: str, request: EmbedQueryRequest):
    log.info(f"v2 compat: /embed-query/{profile_id}")
    return await embed_query(request)