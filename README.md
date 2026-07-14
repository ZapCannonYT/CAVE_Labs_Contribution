# Dr. Aria — Advanced Personal Medical Assistant (Health AI v3.2)

Welcome to the **Health AI v3.2** repository. This project delivers **Dr. Aria**, an offline, stateless personal medical assistant. The backend is built using FastAPI and is designed for highly secure offline execution locally or in production environments.

This documentation serves as a comprehensive guide for developers setting up, running, extending, or verifying the codebase.

---

## 📁 Repository Structure

```tree
├── health_ai/                      # Core backend application
│   ├── api/
│   │   └── server.py               # FastAPI server & endpoint orchestration
│   ├── config/
│   │   └── settings.py             # Server, speed, model path, & folder settings
│   ├── core/
│   │   ├── character.py            # Dr. Aria system prompts, intent classification, & tone mirroring
│   │   ├── exceptions.py           # Custom application exception classes
│   │   ├── logger.py               # Rotating logger engine (max 10MB, 5 backups)
│   │   └── safety.py               # Crisis symptom keyword matching, disclaimer formatting, & inject guards
│   ├── embeddings/
│   │   └── embedder.py             # Singleton interface for BGE embedding model
│   ├── external/
│   │   └── drug_api.py             # RxNorm & DailyMed drug lookup utilities
│   ├── model/
│   │   ├── llm_loader.py           # Singleton LLM loader (supports split & single GGUFs)
│   │   └── PLACE_MODEL_HERE.txt    # Helper file indicating model directory
│   ├── rag/
│   │   ├── chunker.py              # Word-window text chunker (sliding overlap window)
│   │   ├── context_builder.py      # Context assembler & prompt sanitizer
│   │   └── document_processor.py   # Lab values parser & patient history summaries generator
│   ├── tests/
│   │   └── test_ocr_stress.py      # Automated 24-case pipeline stress tests
│   └── utils/
│   │   └── document_reader.py      # Highly robust digital, scanned, or hybrid PDF/Image reader
│   └── logs/                       # Server log output folder
├── chat.html                       # Modern local developer client UI (Popover, Textarea, Animations)
├── requirements.txt                # System requirements & dependencies
├── upgrades_and_improvements.md   # Compilation of architectural improvements in v3.2
└── README.md                       # This developer documentation
```

---

## 🚀 Key Architectural Features & Upgrades

Dr. Aria v3.2 introduces major security, capability, and performance improvements:

### 1. Advanced PDF & OCR Reader Pipeline (`document_reader.py`)
- **Dynamic Routing Classifier:** Automatically inspects uploaded PDF pages and runs them through the optimal path:
  - *Digital Pages (character count > 300):* Instantly parsed via direct text extraction using `pdfplumber`.
  - *Scanned / Image Pages (character count < 50):* Rasterized and run through `PaddleOCR` (with `pytesseract` as fallback).
  - *Mixed Pages (50-300 characters):* Processed using a hybrid extraction and OCR fallback strategy.
- **OCR Quality Guardrails:** 
  - Restricts documents to a maximum of **50 pages** to prevent CPU-intensive locks.
  - Automatically downscales uploaded images exceeding **3000px** on either side using a Lanczos filter.
  - Detects solid color/blank pages to bypass OCR noise, returning a clean empty string rather than triggering scanning errors.
  - Programmatic waiver: Excludes programmatically rasterized PDF pages from camera defects (blur/ lighting/ contrast) tests.

### 2. Smart Hybrid Document Processing (`document_processor.py`)
- **Heuristic Metric Parser:** Extracts patient details, report dates, lab metrics (units like `g/dL`, `mg/dL`, `mmol/L`, `mmHg`), and prescription dosages/schedules (e.g. `1-0-1`, `1-1-1`, `at bedtime`).
- **High-Density Indexing:** Compiles these structured metrics into a high-density summary stored at chunk index 0 (`is_summary: True`). This ensures the LLM has immediate access to key metrics while keeping the remaining text searchable via vector embeddings.

### 3. State-of-the-Art Embedding Model (`embedder.py`)
- Upgraded embedding capabilities to **`BAAI/bge-small-en-v1.5`** to guarantee domain-specific medical query semantic retrieval.
- Automatically prepends recommended search instruction prefixes to queries for optimal embedding space alignment.

### 4. Word-Boundary Intent & Empathy Logic (`character.py`)
- **Whole-Word Matching:** Utilizes word-boundary regex patterns (`\b`) to eliminate false-positive intent classifications.
- **Context-Aware History Fallback:** Allows follow-up questions (e.g. *"why?"*, *"explain further"*, *"thanks!"*) to reach the LLM by checking conversation history, bypassing strict off-topic filters.
- **Dynamic Emotional Mirroring:** Adapts tone to match the patient's context (e.g., cheerful and celebratory for positive recovery updates; warm and reassuring for anxiety/worry; empathetic yet clear for neutral clinical queries).

### 5. Security & Hardening Guardrails
- **DoS File Upload Limits:** Dynamically consumes upload payloads in 64KB chunks; terminates and returns `413 Payload Too Large` immediately if the file size exceeds **25MB** (`server.py`).
- **Non-blocking Operations:** Offloads CPU-intensive PDF parsing/OCR processing to background threads using `asyncio.to_thread` to keep the FastAPI event loop responsive.
- **Log & Prompt Injection Sanitization:** Filters system tokens (like `<|im_start|>`, `<|im_end|>`) in user prompts and document inputs (`context_builder.py`). Sanitizes carriage returns and newlines from filenames before writing to log files to prevent log poisoning.
- **API & Disk Safe Limits:** Enforces a rotating file logger (10MB limits, 5 backup rotation layers) and sets a strict 10-second timeout on drug lookup REST calls.
- **Prompt Bypass Defenses (`safety.py`):** Automatically detects attempts to override system prompts or bypass instructions, returning a standard warm disclaimer: *"I cannot ignore or bypass my instructions to act as a health chatbot. Please ask me a health-related question."*

---

## ⚙️ Setup & Installation

### 1. Prerequisites
Ensure you have **Python 3.10+** installed, along with local OCR dependencies:
- **Tesseract OCR:** Install the Tesseract binaries on your system and add them to your environment PATH.

### 2. Create Virtual Environment
Set up a python virtual environment in the project directory:
```bash
python -m venv .venv

# On Windows (PowerShell):
.\.venv\Scripts\Activate.ps1

# On macOS/Linux:
source .venv/bin/activate
```

### 3. Install Dependencies
Install all required packages:
```bash
pip install -r requirements.txt
```

### 4. Setup LLM Model
1. Download a Qwen2.5 Instruct GGUF model (e.g. `Qwen2.5-14B-Instruct-Q5_K_M.gguf` or a smaller shard model).
2. Place the GGUF file(s) in: `health_ai/model/`
3. Configure the active filename in `health_ai/config/settings.py` via `LLM_MODEL_PATH`.
   - *Note: If using split shards, point the model path to shard 1 (ending in `-00001-of-00003.gguf`). The server automatically detects and loads the remaining shards.*

---

## 🔐 Configuration & Authentication

The FastAPI server enforces authentication on key endpoints (`/upload-and-embed`, `/embed-query`, and `/generate`). 

### 1. Local Development Mode (Bypass Auth)
To test the chatbot locally using the developer UI (`chat.html`) without API keys:
1. Create a `.env` file in the root of the project.
2. Add the following line:
   ```env
   HEALTH_AI_ALLOW_UNAUTHENTICATED=true
   HEALTH_AI_GPU_LAYERS=0
   ```
   *(Set `HEALTH_AI_GPU_LAYERS=-1` to enable full GPU offload if your system has CUDA support).*

### 2. Production Mode (Strict Auth)
In production, clients must supply credentials:
- **X-API-Key:** Matches the `DIGITAL_TWIN_API_KEY` configured in the server environment.
- **Authorization Bearer Token:** Valid Firebase ID Token verifying the user request.

---

## 🚀 Running the Server & Client UI

### 1. Start the Backend Server
Run the FastAPI application locally:
```bash
python -m uvicorn health_ai.api.server:app --host 127.0.0.1 --port 8000 --reload
```
Once loaded, you can view the automated OpenAPI documentation at: `http://127.0.0.1:8000/docs`

### 2. Launch the Developer UI
Double-click or open `chat.html` in any web browser. 
- The client uses modern web features like **HTML Popover API** to display Developer Settings (accessible by clicking the gear icon).
- Autoresizes user input queries dynamically using the CSS **`field-sizing: content`** property.
- Features smooth fade-in and slide-up animations for new message entries using native CSS **`@starting-style`** declarations.

---

## 🧪 Verification & Automated Testing

Verify your setup by running the test suites:

### 1. Unit & Functional Tests
Verify intent classification and context-aware fallback logic:
```bash
python -m pytest health_ai/tests/
```

### 2. Pipeline Stress Tests
Run the robust **24-case OCR and Document Reader stress test suite** which simulates corrupted files, skewed PDFs, decompression bombs, table parsing, low-contrast scans, rotated documents, and high-concurrency requests:
```bash
python health_ai/tests/test_ocr_stress.py
```
*(All 24 stress test cases must pass cleanly to verify OCR guardrails).*
