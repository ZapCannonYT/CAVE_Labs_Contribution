"""
document_reader.py — Text extraction from PDFs and images for Health AI v3.

OCR pipeline (images):
    1. PaddleOCR  — primary, handles angle correction, confidence filtering
    2. pytesseract — fallback, PSM 6 (uniform block of text)
    3. Raises OCRError if both fail

PDF extraction:
    pdfplumber — table-aware PDF text extraction

Confidence threshold: 0.40 (keeps handwritten and low-contrast prescription text)
"""

import logging
from pathlib import Path
from threading import Lock
from typing import Optional

from health_ai.config.settings import OCR_MIN_CONFIDENCE, OCR_MIN_LINE_CHARS
from health_ai.core.logger import get_logger
from health_ai.core.exceptions import OCRError, UnsupportedFileTypeError, EmptyDocumentError

log = get_logger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

SUPPORTED_PDF_EXTENSIONS  = {".pdf"}
SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
SUPPORTED_EXTENSIONS = SUPPORTED_PDF_EXTENSIONS | SUPPORTED_IMAGE_EXTENSIONS

# Lines from lab report headers/footers to strip
_FOOTER_FRAGMENTS = [
    "dr.tejaswini", "dr. sanjeev", "dr.yash", "dr. purvish",
    "dr. hardik", "dr. siddharth", "m.d. pathology", "md path",
    "hematopathologist", "electronically authenticated", "referred test",
    "sterling accuris", "national reference laboratory",
    "b/s. jalaram", "email:", "page ",
]


# ── DocumentReader ─────────────────────────────────────────────────────────────

class DocumentReader:
    """
    Singleton text extractor for PDFs and images.

    Usage:
        reader = DocumentReader()
        text = reader.extract(file_path)
    """

    _instance = None
    _lock = Lock()
    _paddle_lock = Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._paddle = None
                    cls._instance._paddle_attempted = False
        return cls._instance

    # ── PaddleOCR (lazy load) ─────────────────────────────────────────────────

    def _get_paddle(self):
        if self._paddle_attempted:
            return self._paddle
        with self._paddle_lock:
            if not self._paddle_attempted:
                self._paddle_attempted = True
                try:
                    from paddleocr import PaddleOCR
                    self._paddle = PaddleOCR(
                        use_angle_cls=True,
                        lang="en",
                        show_log=False,
                    )
                    log.info("PaddleOCR loaded successfully.")
                except Exception as e:
                    log.warning(f"PaddleOCR unavailable, will use pytesseract fallback: {e}")
                    self._paddle = None
        return self._paddle

    # ── PDF extraction ────────────────────────────────────────────────────────

    def _extract_pdf(self, path: Path) -> str:
        try:
            import pdfplumber
        except ImportError:
            raise OCRError("pdfplumber is not installed. Run: pip install pdfplumber")

        pages_text = []
        try:
            with pdfplumber.open(str(path)) as pdf:
                for page_num, page in enumerate(pdf.pages, start=1):
                    text = page.extract_text() or ""
                    text = self._clean_lab_page(text)
                    if text.strip():
                        pages_text.append(f"[Page {page_num}]\n{text}")
        except Exception as e:
            raise OCRError(f"pdfplumber failed on {path.name}: {e}") from e

        combined = "\n\n".join(pages_text).strip()
        if not combined:
            raise EmptyDocumentError(f"PDF '{path.name}' produced no text after extraction.")
        return combined

    def _clean_lab_page(self, raw: str) -> str:
        """Strip header noise and known footer lines from lab report pages."""
        lines = raw.split("\n")
        cleaned = []
        for line in lines:
            s = line.strip()
            if not s or len(s) < 3:
                continue
            if any(f in s.lower() for f in _FOOTER_FRAGMENTS):
                continue
            cleaned.append(s)
        return "\n".join(cleaned).strip()

    # ── Image OCR — PaddleOCR ─────────────────────────────────────────────────

    def _extract_image_paddle(self, path: Path) -> Optional[str]:
        paddle = self._get_paddle()
        if paddle is None:
            return None

        try:
            result = paddle.ocr(str(path), cls=True)
        except Exception as e:
            log.warning(f"PaddleOCR failed on {path.name}: {e}")
            return None

        if not result or not result[0]:
            return None

        lines = []
        low_conf_lines = []

        for line in result[0]:
            if not line or len(line) < 2:
                continue
            text_conf = line[1]
            if not text_conf or len(text_conf) < 2:
                continue
            text, conf = text_conf[0], text_conf[1]
            text = (text or "").strip()
            if len(text) < OCR_MIN_LINE_CHARS:
                continue
            if conf >= OCR_MIN_CONFIDENCE:
                lines.append(text)
            else:
                low_conf_lines.append(text)   # keep low-confidence too, appended at end

        if not lines and not low_conf_lines:
            return None

        # Put high-confidence lines first, then low-confidence (for prescriptions)
        all_lines = lines + (low_conf_lines if not lines else [])
        return "\n".join(all_lines).strip()

    # ── Image OCR — pytesseract fallback ─────────────────────────────────────

    def _extract_image_tesseract(self, path: Path) -> Optional[str]:
        try:
            import pytesseract
            from PIL import Image
            img = Image.open(str(path))
            # PSM 6 = assume uniform block of text (works well for prescriptions)
            text = pytesseract.image_to_string(img, config="--psm 6")
            text = text.strip()
            return text if text else None
        except Exception as e:
            log.warning(f"pytesseract failed on {path.name}: {e}")
            return None

    def _extract_image(self, path: Path) -> str:
        text = self._extract_image_paddle(path)
        if text:
            log.debug(f"PaddleOCR succeeded for {path.name} ({len(text)} chars)")
            return text

        log.info(f"PaddleOCR returned empty, trying pytesseract for {path.name}")
        text = self._extract_image_tesseract(path)
        if text:
            log.debug(f"pytesseract succeeded for {path.name} ({len(text)} chars)")
            return text

        raise OCRError(
            f"All OCR engines failed for '{path.name}'. "
            "Ensure the image is clear, well-lit, and at least 300 DPI."
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def extract(self, file_path: str) -> str:
        """
        Extract text from a file (PDF or image).

        Args:
            file_path: Absolute or relative path to the file.

        Returns:
            Extracted text as a string.

        Raises:
            UnsupportedFileTypeError: if the extension is not supported.
            OCRError: if image OCR fails.
            EmptyDocumentError: if the file produces no text.
        """
        path = Path(file_path)
        ext = path.suffix.lower()

        if ext not in SUPPORTED_EXTENSIONS:
            raise UnsupportedFileTypeError(
                f"Unsupported file type '{ext}'. "
                f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            )

        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        log.info(f"Extracting text from: {path.name} ({ext})")

        if ext in SUPPORTED_PDF_EXTENSIONS:
            return self._extract_pdf(path)
        else:
            return self._extract_image(path)

    def detect_doc_type(self, filename: str) -> str:
        """
        Guess document type from filename.
        Returns 'prescription' or 'lab_report'.
        """
        name = filename.lower()
        if any(k in name for k in ["presc", "rx", "medicine", "tablet", "dr_"]):
            return "prescription"
        if any(k in name for k in ["blood", "lab", "report", "result", "cbc", "test"]):
            return "lab_report"
        ext = Path(filename).suffix.lower()
        # Images without clear names are usually prescriptions
        return "prescription" if ext in SUPPORTED_IMAGE_EXTENSIONS else "lab_report"
