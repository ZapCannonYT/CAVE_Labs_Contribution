# 🚀 Health AI Upgrades & Architectural Report

This document records the chronological development and implementation of all architectural enhancements, safety guardrails, model upgrades, and optimizations integrated into **`healthbot_v3.2`** relative to the standard **`healthbot_v3.1`** baseline.

---

## 📁 Chronological Compilation of Upgrades

### 1. Repository Configuration & Gitignore Optimization
- **Description:** Set up workspace git policies and structured exclusions to separate core code from local models and backups.
- **Key Deliverables:** Updated [.gitignore](file:///c:/Users/Zap/UHI_Internship/Health-Digital-Twin-Sunav/healthbot_v3.2/.gitignore) to exclude large model weights (`*.gguf`), local compressed archives (`*.zip`), virtual environment directories (`.venv/`), and compilation caches (`__pycache__/`).
- **Technical Impact:** Prevents repository bloat, ensuring that only source code and test files are committed, keeping push payloads light.

---

### 2. Semantic Search & Dense Embedding Upgrade
- **Description:** Upgraded the embedding model to improve matching density and context retrieval for clinical and medical terms.
- **Key Deliverables:**
  - Integrated the state-of-the-art **`BAAI/bge-small-en-v1.5`** embedding model inside [embedder.py](file:///c:/Users/Zap/UHI_Internship/Health-Digital-Twin-Sunav/healthbot_v3.2/health_ai/embeddings/embedder.py).
  - Configured a query instruction prefix: `"Represent this sentence for searching relevant passages: "` prepended to clinical searches.
- **Technical Impact:** Replaced the legacy `all-MiniLM-L6-v2` model, yielding a significant boost in semantic retrieval accuracy within specialized medical query contexts.

---

### 3. Word-Boundary Intent Filtering & Empathy Prompting
- **Description:** Hardened the classifier against false-positive triggers and updated response empathy guidelines.
- **Key Deliverables:**
  - Implemented whole-word boundary regex patterns (`\b`) inside [character.py](file:///c:/Users/Zap/UHI_Internship/Health-Digital-Twin-Sunav/healthbot_v3.2/health_ai/core/character.py) for clinical classifications.
  - Upgraded prompt matrices to mirror patient cues—providing warm reassurance for anxiety/worry and celebratory support for positive clinical developments.
- **Technical Impact:** Prioritizes clinical symptom categorization over generic greetings (resolving false-positives such as *"hello, my chest tightness is back"* being misclassified as a greeting).

---

### 4. Context-Aware Chat History Fallback Router
- **Description:** Restructured the intent router to tolerate conversation context, preventing incorrect prompt blocks on conversational follow-ups.
- **Key Deliverables:** Added fallback history checking inside [character.py](file:///c:/Users/Zap/UHI_Internship/Health-Digital-Twin-Sunav/healthbot_v3.2/health_ai/core/character.py). If a user's prompt is a short follow-up (e.g. *"explain that further"*, *"why?"*, *"thanks"*), it scans preceding messages to check if they were on-topic.
- **Technical Impact:** Resolves strict off-topic rejection failures by allowing relevant follow-ups to reach the LLM rather than blocking them.

---

### 5. PDF classifier & Hybrid Page Parser
- **Description:** Implemented a page-by-page file analyzer to select the optimal text extraction strategy based on document type.
- **Key Deliverables:** Reconstructed [document_reader.py](file:///c:/Users/Zap/UHI_Internship/Health-Digital-Twin-Sunav/healthbot_v3.2/health_ai/utils/document_reader.py) to inspect page character density:
  - *Digital Pages (characters > 300):* Direct extraction via `pdfplumber`.
  - *Scanned Pages (characters < 50):* Page rasterization and OCR processing via `PaddleOCR` (with `pytesseract` fallback).
  - *Mixed Pages (50–300 characters):* Combined extraction with OCR fallback.
- **Technical Impact:** Dramatically reduces processing overhead by extracting digital text directly instead of running expensive OCR on programmatic PDF pages.

---

### 6. OCR Quality Guardrails & Performance Shields
- **Description:** Configured quality validation rules to prevent server locks or visual parsing failures from corrupted or malformed documents.
- **Key Deliverables:**
  - Enforced a hard **50-page document ceiling** to mitigate denial-of-service (DoS) locks.
  - Added Lanczos filtering in [document_reader.py](file:///c:/Users/Zap/UHI_Internship/Health-Digital-Twin-Sunav/healthbot_v3.2/health_ai/utils/document_reader.py) to downscale images exceeding **3000px**.
  - Built blank-page waiving: solid colors (std dev < 1.0) skip quality guards to return a clean empty string.
  - PDF-Rendered Page Waiver: Excludes programmatically rasterized pages from contrast, blur, and lighting tests.
- **Technical Impact:** Stabilizes document uploading against malformed or massive PDFs, preventing memory overflow.

---

### 7. Structured Parameter Extraction Heuristics
- **Description:** Created a parser to extract clinical indicators, dosages, and patient names, indexing them for rapid search.
- **Key Deliverables:**
  - Added [document_processor.py](file:///c:/Users/Zap/UHI_Internship/Health-Digital-Twin-Sunav/healthbot_v3.2/health_ai/rag/document_processor.py) to extract metric patterns (e.g., units like `mg/dL`, `mmol/L`, `bpm`, `mmHg`) and prescription schedules (`1-0-1`).
  - Compiled structured extractions into a high-density Markdown summary placed at chunk index 0 (`is_summary: True`).
- **Technical Impact:** Provides the LLM with immediate access to tabular clinical stats at query time, supplementing raw semantic vector searches.

---

### 8. Prompt Injection Defenses & Global Emoji Scrubber
- **Description:** Hardened prompt templates against system override attacks and stripped emojis to preserve clinical tone.
- **Key Deliverables:**
  - Integrated `sanitize_prompt_input` inside [context_builder.py](file:///c:/Users/Zap/UHI_Internship/Health-Digital-Twin-Sunav/healthbot_v3.2/health_ai/rag/context_builder.py) to strip system tokens (e.g., `<|im_start|>`, `<|im_end|>`).
  - Implemented regex-based keyword safety checks inside [safety.py](file:///c:/Users/Zap/UHI_Internship/Health-Digital-Twin-Sunav/healthbot_v3.2/health_ai/core/safety.py) to detect override attempts.
  - Built a global regex emoji scrubber inside `apply_safety_layer` to strip emojis from the final output text.
- **Technical Impact:** Restricts conversational tone to clinical boundaries and blocks adversarial injection payloads.

---

### 9. Non-Blocking Server Threads & File Size Limits
- **Description:** Hardened the API interface to handle upload payloads safely without blocking the web event loop.
- **Key Deliverables:**
  - Added stream size verification in [server.py](file:///c:/Users/Zap/UHI_Internship/Health-Digital-Twin-Sunav/healthbot_v3.2/health_ai/api/server.py)'s `/upload-and-embed` endpoint: rejects uploads exceeding **25MB** immediately.
  - Offloaded CPU-heavy extraction and OCR to background executor threads via `asyncio.to_thread`.
- **Technical Impact:** Prevents large files from starving the server's thread pool, ensuring concurrent clients remain active.

---

### 10. Developer Playground UI Upgrades (`chat.html`)
- **Description:** Overhauled the web testing interface for a premium, high-fidelity experience.
- **Key Deliverables:**
  - Redesigned the sidebar file-upload layout, styling the "Upload File" and "Cancel" actions side-by-side in secondary low-contrast designs.
  - Added a shimmery animated progress bar that cycles through extraction and OCR status stages.
  - Utilized CSS `field-sizing: content` to automatically resize the chat input textarea.
  - Added CSS `@starting-style` declarations to introduce smooth slide-in entry animations for new message bubbles.
- **Technical Impact:** Elevates the local playground UI to a premium level, making tests visually descriptive and easy to interact with.

---

### 11. Malicious & Edge-Case Stress Test Suite
- **Description:** Implemented an end-to-end stress test suite to validate OCR stability and exception bubbling.
- **Key Deliverables:** Developed [test_ocr_stress.py](file:///c:/Users/Zap/UHI_Internship/Health-Digital-Twin-Sunav/healthbot_v3.2/health_ai/tests/test_ocr_stress.py) with 24 edge-case tests, verifying correct error bubbling for empty, password-protected, js-embedded, and high-concurrency documents.
- **Technical Impact:** Guarantees code regression stability under extreme or malformed inputs.

---

### 12. Architectural Developer Onboarding Documentation
- **Description:** Restructured technical documents to aid developer onboarding.
- **Key Deliverables:** Rewrote [README.md](file:///c:/Users/Zap/UHI_Internship/Health-Digital-Twin-Sunav/healthbot_v3.2/README.md) to serve as a clean technical guide, featuring system package layouts, visual Mermaid flowcharts, setup instructions, and Windows security bypass tips.
- **Technical Impact:** Eliminates onboarding friction by separating core architecture details from project histories.

---

### 13. Dynamic Model Shard Auto-Verification
- **Description:** Enabled the engine to load single-file model weights dynamically while maintaining shard verification for multi-file models.
- **Key Deliverables:** Re-engineered the loader inside [llm_loader.py](file:///c:/Users/Zap/UHI_Internship/Health-Digital-Twin-Sunav/healthbot_v3.2/health_ai/model/llm_loader.py) to check for split model file shards *only* when the primary filename ends with `"00001-of-00003"`.
- **Technical Impact:** Allows single-file models (like `Qwen3-4B-Q4_K_M.gguf`) to bypass shard checks and start up cleanly, resolving the startup crash.

---

### 14. Streaming Generator Disconnect Watcher
- **Description:** Implemented a cancellation listener to abort text generation immediately if a client disconnects.
- **Key Deliverables:** Updated the `generate()` method in [llm_loader.py](file:///c:/Users/Zap/UHI_Internship/Health-Digital-Twin-Sunav/healthbot_v3.2/health_ai/model/llm_loader.py) to accept `stop_event` and `meta` parameters, invoking `stream=True` on `Llama` to yield tokens chunk-by-chunk and aborting if `stop_event.is_set()` is detected.
- **Technical Impact:** Resolves uvicorn `TypeError` crashes and frees CPU resources immediately upon connection disconnects during generation.

---

### 15. Workspace Developer Custom Skills
- **Description:** Installed developer-focused workspace skills to enforce visual quality across project deliverables.
- **Key Deliverables:** Created a custom workspace formatting skill at [SKILL.md](file:///c:/Users/Zap/UHI_Internship/Health-Digital-Twin-Sunav/healthbot_v3.2/.agents/skills/prettier_markdown/SKILL.md) outlining strict rules for clean typography, Mermaid logic diagrams, GitHub-style alerts, and reduced/minimal emoji usage.
- **Technical Impact:** Guarantees that future AI-generated documentation remains highly structured, clean, and professional.
