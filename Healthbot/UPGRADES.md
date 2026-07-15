# Health AI Upgrades & Architectural Report

This document records the step-by-step development process and implementation of all architectural enhancements, safety guardrails, model upgrades, and optimizations integrated into the Health AI v3.2 system compared to the standard v3.1 baseline.

---

## Chronological Compilation of Upgrades

### Repository Tracking & Exclusion Rules
- **Workspace Hygiene:** Configured clean file tracking constraints to isolate development files, dependencies, and local artifacts from version control.
- **Dependency Isolation:** Formulated exclusions in `.gitignore` to reject virtual environment directories (`.venv/`), metadata caches (`__pycache__/`, `.pytest_cache/`), and IDE configurations (`.idea/`, `.vscode/`).
- **Large File Constraints:** Explicitly blacklisted model weights (`*.gguf`) and backup archives (`*.zip`) from Git index staging. This prevents accidental pushes of multi-gigabyte models and complies with Git remote repository size limits.
- **Tracking Integrity:** Confirmed that all relevant test fixtures, backend python packages, HTML UI pages, and configuration files are staged cleanly.

---

### Dense Semantic Search & Embedding Upgrade
- **Model Migration:** Upgraded the system's vector representation engine from `all-MiniLM-L6-v2` to the advanced **`BAAI/bge-small-en-v1.5`** dense embedding model.
- **Dimensionality Matching:** Configured the embedding singleton in `health_ai/embeddings/embedder.py` to map document segments into a dense 384-dimensional space.
- **Search Query Prefixes:** Implemented recommended search prefix instructions. Every user query embedding request automatically prepends: `"Represent this sentence for searching relevant passages: "`.
- **Retrieval Performance:** Dramatically improved cosine similarity matching density for clinical medical terms, diagnostics, and patient report segments.

---

### Intent Routing & Dynamic Empathy Prompting
- **Whole-Word Matching:** Swapped simple substring matches for strict whole-word boundary regex patterns (`\b`) inside `health_ai/core/character.py`. This prevents overlapping keywords from triggering incorrect classifications.
- **Intent Priority mapping:** Rewrote classifier routing rules to prioritize clinical/medical symptom detection over generic greetings (e.g. *"hello, my chest tightness is back"* correctly routes to the Symptom template instead of the Greeting greeting block).
- **Dynamic Emotional Mirroring:** Configured system prompt templates to adapt the chatbot's tone to match patient cues. The engine dynamically chooses high warmth and reassurance for anxiety/pain, and celebratory support for positive lab indicators or recovery updates.
- **Clinical Explanations:** Prompts require the LLM to outline responses in plain, layperson-friendly terms, avoiding jargon and structuring recommendations with headers.

---

### Context-Aware Conversation Fallback Router
- **Rejection Mitigation:** Hardened the off-topic prompt filter to prevent short, conversational follow-up questions from being incorrectly rejected.
- **History Checking:** Built a fallback history scanner inside `health_ai/core/character.py`. If a user's prompt is categorized as off-topic, the router reviews the thread's recent conversation logs.
- **Context Preservation:** If the immediately preceding discussion was medical or on-topic, the short prompt (such as *"why?"*, *"explain more"*, or *"thanks!"*) is passed through to the LLM context.
- **Conversational Smoothness:** Eliminates rigid chatbot blocking, enabling natural conversational flow during patient Q&A.

---

### Dynamic Document Extraction Classifier
- **Resource Routing:** Designed a page-by-page file analyzer in `health_ai/utils/document_reader.py` that inspects each PDF page to choose the most efficient text extraction strategy.
- **Digital Page Parsing:** Digital PDF pages containing selectable text (character count > 300) are routed directly to `pdfplumber` for instant extraction.
- **Scanned Image Processing:** Scanned image pages (character count < 50) are rasterized into images and routed to `PaddleOCR` (falling back to `pytesseract` if needed).
- **Hybrid Mixed Strategy:** Mixed pages (50-300 characters) try digital text extraction first and fall back to image-based OCR only if the extracted character count remains low.
- **Execution Speed:** Reduces parsing time by over 90% by bypassing expensive OCR workloads for programmatic PDF pages.

---

### OCR Quality Guardrails & Performance Shields
- **Resource Protection:** Established quality checks inside `document_reader.py` to prevent CPU locks or server timeouts during document uploads.
- **Image Scaling Thresholds:** Uploaded images or page raster images exceeding **3000px** on either side are automatically downscaled using a Lanczos filter.
- **Page Ceilings:** Restricts incoming documents to a maximum of **50 pages** per PDF, returning a clean error if exceeded.
- **Blank Page Bypass:** solid-color pages (contrast standard deviation < 1.0) bypass blur and lighting tests, returning an empty string instead of triggering OCR errors.
- **Rendered PDF Waivers:** Programmatically rendered pages are exempted from blur, contrast, and exposure checks since they do not contain camera defects.

---

### Heuristic Clinical Parameter Summarization
- **Metric Extraction Heuristics:** Built `health_ai/rag/document_processor.py` to parse medical reports and identify names, report dates, clinical metrics, and dosages.
- **Parametric Regex Matching:** Scans text for units (e.g. `g/dL`, `mg/dL`, `mmol/L`, `mmHg`) and prescription frequencies (e.g., `1-0-1`, `1-1-1`, `at bedtime`).
- **High-Density Indexing:** Compiles these structured metrics into a dense Markdown summary placed at chunk index 0 (`is_summary: True`).
- **Context Inject Priority:** The summary chunk is injected first into the LLM context, ensuring the assistant has immediate access to core stats alongside standard vector embeddings.

---

### Prompt Injection & Log Spoofing Defenses
- **System Token Scrubbing:** Implemented `sanitize_prompt_input` inside `health_ai/rag/context_builder.py` to strip out system/chat markers (e.g., `<|im_start|>`, `<|im_end|>`) from user inputs and document segments.
- **Override Protections:** Integrated regex safeguards (`_INJECTION_REGEX`) inside `health_ai/core/safety.py` and a short-circuit bypass filter in `server.py` to detect prompts attempting to bypass instructions (e.g. *"ignore all instructions"*), immediately returning `BYPASS_ATTEMPT_RESPONSE` directly.
- **Filename Sanitization:** Sanitizes newlines and carriage returns (`\r`, `\n`) from incoming filenames before logging to prevent log spoofing/injection.
- **Hotline Alerts:** Enforces strict symptom keywords checks to append urgent medical contact details (911 / 999 / 112) to responses when emergency indicators are detected.

---

### Non-Blocking API Offloading & Size Bounds
- **Chunked Payload Validation:** Overhauled the `/upload-and-embed` endpoint in `health_ai/api/server.py` to consume payloads in 64KB chunks, immediately raising a `413 Payload Too Large` if the file size exceeds **25MB**.
- **FastAPI Event Loop Safety:** Offloaded CPU-heavy extraction and OCR workloads to background threads using `asyncio.to_thread`, keeping the async event loop responsive.
- **API Call Timeouts:** Implemented a strict 10-second timeout on all external REST lookups in `health_ai/external/drug_api.py`.
- **Disk Space Preservation:** Configured a `RotatingFileHandler` inside `health_ai/core/logger.py` capped at 10MB with a 5-file rotation limit to prevent logs from filling local storage.

---

### High-Fidelity Developer Playground UI
- **Action Button Styling:** Redesigned the sidebar upload interface in `chat.html` to display "Upload File" and "Cancel" side-by-side using secondary low-contrast borders.
- **Progress Animation:** Added a shimmery loading bar that cycles through document loading, text extraction, and indexing phases.
- **Dynamic Text Input:** Utilized CSS `field-sizing: content` on the chat text box to automatically expand height as the developer types.
- **Visual Entries:** Configured CSS `@starting-style` entry transitions, introducing smooth slide-in animations for chat bubble rendering.

---

### Malicious & Edge-Case Stress Testing
- **Robust Verification Suite:** Developed [test_ocr_stress.py](health_ai/tests/test_ocr_stress.py) to validate the document reader pipeline against hostile or corrupted documents.
- **Stress Scenarios:** Verifies correct exception handling and error bubbling for malformed files, password-protected PDFs, javascript-embedded PDFs, and high-concurrency requests.
- **Edge-Case Mocking:** Features 24 test cases simulating blurred scans, rotated blocks, multi-column tables, skewed photos, empty documents, and non-English scripts.
- **Regression Guard:** Ensures OCR pipeline changes can be verified locally for safety before staging commits.

---

### Dynamic Model Shard Auto-Verification
- **Weight Check Decoupling:** Re-engineered the LLM engine startup inside `health_ai/model/llm_loader.py` to verify GGUF model splits dynamically.
- **Decoupled Verification:** Shard verification (checking for `00002-of-00003`, etc.) is executed *only* when the primary path filename contains `"00001-of-00003"`.
- **Model Flexibility:** Allows single-file models (like `Qwen3-30B-A3B.gguf`) to bypass shard checks and load cleanly, resolving startup crash loops.

---

### Stream-Disconnect LLM Generator Abort Watcher
- **FastAPI Endpoint Sync:** Updated the `generate()` method signature in `llm_loader.py` to accept `stop_event` and `meta` dictionary arguments.
- **Mid-Generation Cancellation:** Configured `stream=True` on `Llama` generation. The loop evaluates the `stop_event` thread flag during token yields, immediately aborting the text generation if the client disconnects.
- **Duration Instrumentation:** Tracks exact generation time by writing start timestamps to `meta["t_start"]` immediately upon lock acquisition, logging generation duration.

---

### Workspace Layout & Boilerplate Restoration
- **Workspace Hygiene:** Restructured the repository layout into a clean monorepo, keeping all backend packages organized inside the `Healthbot/` subdirectory (with sub-modules under `Healthbot/health_ai/`) to prevent root clutter.
- **Boilerplate Recovery:** Restored the 13 essential backend boilerplate files from the local backup archives without overwriting existing custom updates or model settings.
- **Safety Testing Coverage:** Added unit test coverage in `tests/test_safety.py` to validate both red-flag medical emergencies and prompt-injection override detection.
