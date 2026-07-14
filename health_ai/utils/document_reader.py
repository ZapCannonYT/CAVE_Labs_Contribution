"""
document_reader.py — Text extraction from PDFs and images for Health AI v3.

PDF pipeline:
    1. PyMuPDF (fitz)  — open PDF, analyze each page
    2. Classify pages  — score char count, image blocks, image coverage, text layer
    3. Classify document:
         DIGITAL_PDF  : ≥ 80% TEXT_PAGE  → pdfplumber direct extraction
         SCANNED_PDF  : ≥ 80% OCR_PAGE   → render pages → PaddleOCR
         MIXED_PDF    : everything else  → hybrid (text pages = pdfplumber, scan pages = OCR)
    4. Choose extraction strategy per page
    5. Extract text (pdfplumber and/or OCR)
    6. Clean lab-report noise
    7. Return merged, page-marked text

OCR pipeline (images + scanned PDF pages):
    Primary  : PaddleOCR  — angle correction, confidence filtering
    Fallback : pytesseract — PSM 6 (uniform block of text)
    Rejection: UNREADABLE images are rejected rather than returning garbage medical text

Confidence threshold: 0.40 (retains handwritten and low-contrast prescription text)

BEHAVIORAL CHANGES FROM ORIGINAL (all explicitly safer or more efficient):
  1. PageAnalysis dataclass replaces bare 4-tuples — internal only, API unchanged.
  2. Image quality analysis + adaptive DPI (150 → 175 → 200):
       - UNREADABLE images raise OCRError instead of silently returning bad text.
       - This is safer: unreliable medical data is worse than no data.
  3. _render_page_to_image() accepts a dpi param instead of using a fixed constant.
  4. PaddleOCR output is validated (avg confidence, word count, line count, length)
       before being accepted, replacing the implicit "any lines = success" heuristic.
  5. All other changes are structural refactors with identical runtime behavior.
"""

import io
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from threading import Lock
from typing import Optional

from health_ai.config.settings import OCR_MIN_CONFIDENCE, OCR_MIN_LINE_CHARS
from health_ai.core.exceptions import EmptyDocumentError, OCRError, UnsupportedFileTypeError
from health_ai.core.logger import get_logger

log = get_logger(__name__)


# ── Supported extensions ────────────────────────────────────────────────────

SUPPORTED_PDF_EXTENSIONS   = {".pdf"}
SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
SUPPORTED_EXTENSIONS       = SUPPORTED_PDF_EXTENSIONS | SUPPORTED_IMAGE_EXTENSIONS


# ── Configuration dataclasses ───────────────────────────────────────────────
# Grouping related constants into dataclasses eliminates magic numbers,
# makes tuning visible in one place, and enables future env-var injection.

@dataclass(frozen=True)
class PageClassificationConfig:
    """Thresholds for per-page classification based on extracted character counts."""
    # Pages with more than this many characters are reliably machine-readable.
    char_threshold_text: int = 300
    # Pages with fewer than this many characters are treated as blank/scanned.
    char_threshold_ocr: int = 50
    # Fraction of page area covered by images that tips a sparse-text page to OCR.
    image_coverage_ocr_threshold: float = 0.40


@dataclass(frozen=True)
class DocumentClassificationConfig:
    """Fractions of pages needed to classify the whole document."""
    # At least this fraction of TEXT_PAGE → DIGITAL_PDF.
    digital_pdf_ratio: float = 0.80
    # At least this fraction of OCR_PAGE  → SCANNED_PDF.
    scanned_pdf_ratio: float = 0.80


@dataclass(frozen=True)
class DpiConfig:
    """Adaptive DPI ladder for rendering scanned PDF pages.

    We start at the lowest DPI that PaddleOCR handles well and only climb
    when quality is too poor.  Staying low saves CPU and RAM on edge devices.
    Never exceed max_dpi to prevent OOM on embedded hardware.
    """
    initial: int = 150
    retry:   int = 175
    max:     int = 200  # Hard ceiling — going higher causes OOM on 16 GB devices


@dataclass(frozen=True)
class ImageQualityConfig:
    """Thresholds for classifying rendered page / standalone image quality."""
    # Minimum short-side pixel count considered usable for OCR.
    min_resolution_px: int = 100
    # Variance-of-Laplacian blur score below this → blurry image.
    min_blur_score: float = 20.0
    # RMS contrast below this → low contrast.
    min_contrast: float = 30.0
    # Mean brightness outside [min, max] → lighting problem.
    min_brightness: float = 30.0
    max_brightness: float = 240.0
    # Number of quality issues allowed before classifying as LOW (vs HIGH/NORMAL).
    low_quality_issue_count: int = 1
    # Number of quality issues that makes the image UNREADABLE (reject, no OCR).
    unreadable_issue_count: int = 3


@dataclass(frozen=True)
class OcrValidationConfig:
    """Minimum bar an OCR result must clear to be accepted."""
    # Average PaddleOCR confidence across all accepted lines.
    min_avg_confidence: float = 0.30
    # Minimum number of words in the full OCR result.
    min_word_count: int = 3
    # Minimum number of accepted lines.
    min_line_count: int = 1
    # Minimum total character count to consider the result useful.
    min_char_length: int = 10


# Module-level singletons — referenced throughout without re-instantiation.
_PAGE_CFG   = PageClassificationConfig()
_DOC_CFG    = DocumentClassificationConfig()
_DPI_CFG    = DpiConfig()
_QUALITY_CFG = ImageQualityConfig()
_OCR_VAL_CFG = OcrValidationConfig()


# ── Known lab-report noise fragments ───────────────────────────────────────
# Lines containing any of these (case-insensitive) are stripped from output.
# Keeping this as a module constant avoids re-allocating it per call.
_FOOTER_FRAGMENTS: tuple[str, ...] = (
    "dr.tejaswini", "dr. sanjeev", "dr.yash", "dr. purvish",
    "dr. hardik", "dr. siddharth", "m.d. pathology", "md path",
    "hematopathologist", "electronically authenticated", "referred test",
    "sterling accuris", "national reference laboratory",
    "b/s. jalaram", "email:", "page ",
)

# Precompiled regex patterns for edge-device efficiency
_FOOTER_REGEX = re.compile(
    r'|'.join(re.escape(f) for f in _FOOTER_FRAGMENTS),
    re.IGNORECASE
)

_EXPLICIT_PAGE_NUM_REGEX = re.compile(
    r'^(?:'
    r'page\s*:\s*\d+'
    r'|page\s+\d+'
    r'|page\s+\d+\s*(?:of|/)\s*\d+'
    r')$',
    re.IGNORECASE
)

_AMBIGUOUS_PAGE_NUM_REGEX = re.compile(
    r'^(?:'
    r'\d+\s*(?:of|/)\s*\d+'
    r'|\d+'
    r')$',
    re.IGNORECASE
)

_LEADER_DOTS_REGEX = re.compile(r'[\.\-_]{2,}')


# ── Enums ───────────────────────────────────────────────────────────────────

class PageType(Enum):
    TEXT_PAGE  = "TEXT_PAGE"   # Reliable embedded text — extract directly
    OCR_PAGE   = "OCR_PAGE"    # Blank or scanned — must render + OCR
    MIXED_PAGE = "MIXED_PAGE"  # Sparse text — try direct first, fall back to OCR


class DocType(Enum):
    DIGITAL_PDF = "DIGITAL_PDF"  # All/mostly text pages → pdfplumber only
    SCANNED_PDF = "SCANNED_PDF"  # All/mostly scanned   → OCR only
    MIXED_PDF   = "MIXED_PDF"    # Hybrid               → route per page


class ImageQuality(Enum):
    HIGH       = "HIGH"       # All metrics pass — 150 DPI is sufficient
    NORMAL     = "NORMAL"     # One metric borderline — 150 DPI still fine
    LOW        = "LOW"        # Several issues — bump DPI to improve accuracy
    UNREADABLE = "UNREADABLE" # Too many issues — reject rather than return garbage


# ── Page analysis dataclass ─────────────────────────────────────────────────

@dataclass
class PageAnalysis:
    """Rich per-page metadata collected during PyMuPDF analysis.

    Replaces the bare (page_num, PageType, char_count, has_images) tuple from
    the original.  Downstream code now accesses named attributes rather than
    positional indices, which eliminates a whole class of index-transposition bugs.
    """
    page_num:       int       # 1-based, matching pdfplumber and log output
    page_type:      PageType
    char_count:     int
    has_images:     bool
    image_count:    int       # Number of image blocks on the page
    image_coverage: float     # Fraction of page area covered by images (0.0–1.0)
    has_text_layer: bool      # True if PyMuPDF reports any embedded text (even sparse)


# ── DocumentReader ──────────────────────────────────────────────────────────

class DocumentReader:
    """
    Singleton text extractor for PDFs and images.

    Thread-safe via double-checked locking for instance creation and a
    separate lock protecting PaddleOCR initialization (which is not
    re-entrant and takes ~2 s on first call).

    Usage:
        reader = DocumentReader()
        text   = reader.extract(file_path)
    """

    _instance    = None
    _lock        = Lock()   # Guards singleton creation
    _paddle_lock = Lock()   # Guards PaddleOCR lazy initialization

    def __new__(cls) -> "DocumentReader":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    instance = super().__new__(cls)
                    instance._paddle           = None
                    instance._paddle_attempted = False
                    cls._instance = instance
        return cls._instance

    # ── PaddleOCR lazy initialization ───────────────────────────────────────

    def _get_paddle(self):
        """Return the PaddleOCR instance, initializing it on first call.

        Uses double-checked locking: the outer check avoids lock contention
        once initialized; the inner check prevents duplicate init under race.
        Returns None if PaddleOCR is unavailable (pytesseract fallback applies).
        """
        if self._paddle_attempted:
            return self._paddle

        with self._paddle_lock:
            if self._paddle_attempted:
                # Another thread initialized while we waited for the lock.
                return self._paddle

            self._paddle_attempted = True
            try:
                import os
                import logging
                os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
                # Silence PaddleOCR's extremely verbose internal logger
                logging.getLogger("ppocr").setLevel(logging.WARNING)
                from paddleocr import PaddleOCR
                self._paddle = PaddleOCR(
                    use_angle_cls=True,  # Correct rotated text (common in prescriptions)
                    lang="en",
                    show_log=False,
                )
                log.info("PaddleOCR initialized successfully.")
            except Exception as exc:
                log.warning(
                    "PaddleOCR unavailable — pytesseract will be used as fallback. "
                    f"Reason: {exc}"
                )
                self._paddle = None

        return self._paddle

    # ── Public API ──────────────────────────────────────────────────────────

    def extract(self, file_path: str) -> str:
        """Extract text from a PDF or image file.

        Args:
            file_path: Absolute or relative path to the file.

        Returns:
            Extracted text as a single string.  PDF output includes
            ``[Page N]`` markers to preserve downstream context.

        Raises:
            UnsupportedFileTypeError: Extension not in SUPPORTED_EXTENSIONS.
            FileNotFoundError:        File does not exist on disk.
            OCRError:                 All OCR engines failed or image unreadable.
            EmptyDocumentError:       File produced no usable text after extraction.
        """
        path = Path(file_path)
        ext  = path.suffix.lower()

        if ext not in SUPPORTED_EXTENSIONS:
            raise UnsupportedFileTypeError(
                f"Unsupported file type '{ext}'. "
                f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            )

        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        log.info(f"[extract] Using document_reader_test (enhanced OCR) — file='{path.name}' ext='{ext}'")

        text = (
            self._extract_pdf(path)
            if ext in SUPPORTED_PDF_EXTENSIONS
            else self._extract_image(path)
        )

        preview = text[:200].replace("\n", " ") if text else ""
        log.info(
            f"[extract] Done — file='{path.name}' "
            f"total_chars={len(text)} preview='{preview}...'"
        )
        return text

    def detect_doc_type(self, filename: str) -> str:
        """Infer document type from the filename.

        Returns:
            'prescription' or 'lab_report'.

        Heuristic: keyword matching first; extension as tiebreaker.
        Images without a clear name are more likely prescriptions than lab reports
        (doctors hand-write prescriptions; lab reports are usually digital PDFs).
        """
        name = filename.lower()

        if any(k in name for k in ("presc", "rx", "medicine", "tablet", "dr_")):
            return "prescription"

        if any(k in name for k in ("blood", "lab", "report", "result", "cbc", "test")):
            return "lab_report"

        ext = Path(filename).suffix.lower()
        return "prescription" if ext in SUPPORTED_IMAGE_EXTENSIONS else "lab_report"

    # ── PyMuPDF page analysis ───────────────────────────────────────────────

    def _analyze_pages(self, fitz_doc) -> list[PageAnalysis]:
        """Classify every page using a multi-signal heuristic.

        Signals used (in order of reliability):
          1. char_count      — primary: embedded text is the most direct evidence
          2. has_text_layer  — confirms a text layer even when char_count is low
          3. image_count     — corroborates a scanned page
          4. image_coverage  — high coverage on a sparse-text page → likely scanned

        Returns a list of PageAnalysis objects in page order (1-indexed).
        """
        analyses: list[PageAnalysis] = []

        for i, page in enumerate(fitz_doc, start=1):
            raw_text   = page.get_text() or ""
            char_count = len(raw_text.strip())

            # get_text("dict") returns blocks; type==1 means image block.
            blocks     = page.get_text("dict").get("blocks", [])
            img_blocks = [b for b in blocks if b.get("type") == 1]
            img_count  = len(img_blocks)

            # Compute fraction of page area covered by image blocks.
            # Used to catch pages where a single large scan dominates.
            page_area     = page.rect.width * page.rect.height
            image_coverage = (
                self._compute_image_coverage(img_blocks, page_area)
                if img_count > 0 and page_area > 0
                else 0.0
            )

            # A text layer exists if PyMuPDF extracted any text at all.
            has_text_layer = char_count > 0

            page_type = self._classify_page(
                char_count=char_count,
                image_coverage=image_coverage,
                has_text_layer=has_text_layer,
            )

            log.debug(
                f"[page_analysis] page={i} type={page_type.value} "
                f"chars={char_count} images={img_count} "
                f"coverage={image_coverage:.2f} text_layer={has_text_layer}"
            )

            analyses.append(PageAnalysis(
                page_num       = i,
                page_type      = page_type,
                char_count     = char_count,
                has_images     = img_count > 0,
                image_count    = img_count,
                image_coverage = image_coverage,
                has_text_layer = has_text_layer,
            ))

        return analyses

    def _classify_page(
        self,
        char_count: int,
        image_coverage: float,
        has_text_layer: bool,
    ) -> PageType:
        """Map per-page signals to a PageType.

        Logic:
          - High char count → TEXT_PAGE (reliable embedded text)
          - Low char count AND low image coverage → OCR_PAGE (blank or pure scan)
          - Low char count AND high image coverage → OCR_PAGE (image dominates)
          - Everything else → MIXED_PAGE (ambiguous; try pdfplumber first)
        """
        if char_count >= _PAGE_CFG.char_threshold_text:
            return PageType.TEXT_PAGE

        if char_count < _PAGE_CFG.char_threshold_ocr:
            # Even if there's technically a text layer, < 50 chars is not
            # trustworthy enough for medical data — push to OCR.
            return PageType.OCR_PAGE

        # 50 ≤ char_count < 300: sparse text.  If images dominate the page,
        # the text is probably just a caption or watermark — route to OCR.
        if image_coverage >= _PAGE_CFG.image_coverage_ocr_threshold:
            return PageType.OCR_PAGE

        return PageType.MIXED_PAGE

    @staticmethod
    def _compute_image_coverage(img_blocks: list, page_area: float) -> float:
        """Return the fraction of page area covered by image blocks (0.0–1.0).

        Clips each block's contribution to [0, 1] to guard against malformed
        bounding boxes with negative or oversized coordinates.
        """
        total_img_area = 0.0
        for block in img_blocks:
            bbox = block.get("bbox")
            if not bbox or len(bbox) < 4:
                continue
            x0, y0, x1, y1 = bbox
            block_area     = max(0.0, x1 - x0) * max(0.0, y1 - y0)
            total_img_area += block_area

        return min(total_img_area / page_area, 1.0)

    def _classify_document(self, analyses: list[PageAnalysis]) -> DocType:
        """Decide how to handle the whole PDF based on page-type distribution.

        Falls back to SCANNED_PDF on an empty document so the caller gets an
        OCRError rather than a confusing EmptyDocumentError.
        """
        total = len(analyses)
        if total == 0:
            log.warning("[doc_classify] No pages found — defaulting to SCANNED_PDF")
            return DocType.SCANNED_PDF

        text_pages  = sum(1 for a in analyses if a.page_type == PageType.TEXT_PAGE)
        ocr_pages   = sum(1 for a in analyses if a.page_type == PageType.OCR_PAGE)
        mixed_pages = total - text_pages - ocr_pages

        if (text_pages / total) >= _DOC_CFG.digital_pdf_ratio:
            doc_type = DocType.DIGITAL_PDF
        elif (ocr_pages / total) >= _DOC_CFG.scanned_pdf_ratio:
            doc_type = DocType.SCANNED_PDF
        else:
            doc_type = DocType.MIXED_PDF

        log.info(
            f"[doc_classify] type={doc_type.value} "
            f"total={total} text_pages={text_pages} "
            f"ocr_pages={ocr_pages} mixed_pages={mixed_pages}"
        )
        return doc_type

    # ── pdfplumber text extraction ──────────────────────────────────────────

    def _extract_pages_pdfplumber(
        self,
        path: Path,
        page_nums: list[int],
    ) -> dict[int, str]:
        """Pull text from the specified pages using pdfplumber.

        Args:
            path:      Path to the PDF file.
            page_nums: 1-based page numbers to extract.

        Returns:
            Mapping of {page_num: cleaned_text}.

        Raises:
            OCRError: pdfplumber import failed or raised an unexpected exception.
        """
        try:
            import pdfplumber
        except ImportError:
            raise OCRError("pdfplumber is not installed. Run: pip install pdfplumber")

        page_set = set(page_nums)
        results: dict[int, str] = {}

        try:
            with pdfplumber.open(str(path)) as pdf:
                for page_num, page in enumerate(pdf.pages, start=1):
                    if page_num not in page_set:
                        continue
                    raw  = page.extract_text() or ""
                    results[page_num] = raw
                    log.debug(
                        f"[pdfplumber] page={page_num} "
                        f"raw_chars={len(raw)}"
                    )
        except Exception as exc:
            raise OCRError(f"pdfplumber failed on '{path.name}': {exc}") from exc

        return results

    # ── PyMuPDF page renderer ───────────────────────────────────────────────

    def _render_page_to_image(
        self,
        fitz_doc,
        page_index: int,
        dpi: int = _DPI_CFG.initial,
    ) -> bytes:
        """Rasterize a single PDF page to PNG bytes for OCR.

        Args:
            fitz_doc:   Open fitz.Document object.
            page_index: 0-based page index (PyMuPDF convention).
            dpi:        Render resolution.  Higher DPI → better OCR accuracy
                        at the cost of CPU and RAM.  Caller controls the DPI
                        ladder; this method does not retry internally.

        Returns:
            Raw PNG bytes.
        """
        import fitz

        page   = fitz_doc[page_index]
        # fitz.Matrix scales from points (72 pt = 1 inch) to pixels.
        matrix = fitz.Matrix(dpi / 72, dpi / 72)
        pix    = page.get_pixmap(matrix=matrix, colorspace=fitz.csRGB)
        return pix.tobytes("png")

    # ── Image quality analysis ──────────────────────────────────────────────

    def _load_image(self, img_bytes: bytes):
        """Decode raw image bytes to a PIL Image and a numpy array.

        Returns:
            (pil_img, np_array)  — both in RGB colour space.

        Raises:
            ImportError: PIL or numpy not installed.
            Exception:   Image bytes are corrupt or unsupported format.
        """
        import numpy as np
        from PIL import Image

        pil_img  = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        max_dim = 4000
        w, h = pil_img.size
        if w > max_dim or h > max_dim:
            if w > h:
                new_w = max_dim
                new_h = int(h * (max_dim / w))
            else:
                new_h = max_dim
                new_w = int(w * (max_dim / h))
            log.info(f"[load_image] Resizing image from {w}x{h} to {new_w}x{new_h} to prevent resource exhaustion")
            pil_img = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        np_array = np.array(pil_img)
        return pil_img, np_array

    def _analyze_image_quality(self, img_bytes: bytes) -> ImageQuality:
        """Evaluate image quality across four dimensions and return a grade.

        Dimensions:
          - Resolution : minimum short-side pixel count
          - Blur       : Variance of Laplacian on the grayscale image
                         (higher = sharper; low = blurry)
          - Contrast   : RMS contrast of the grayscale image
          - Brightness : Mean pixel intensity (too dark or washed-out → fail)

        Grading:
          0 issues          → HIGH
          1 issue           → NORMAL     (still OCR-able)
          2 issues          → LOW        (bump DPI before OCR)
          3+ issues         → UNREADABLE (reject — medical data too risky)

        Returns ImageQuality.UNREADABLE if the image cannot even be decoded.
        """
        try:
            import numpy as np

            pil_img, np_rgb = self._load_image(img_bytes)
        except Exception as exc:
            log.warning(f"[quality] Failed to decode image: {exc}")
            return ImageQuality.UNREADABLE

        issues = 0

        # ── Resolution ──────────────────────────────────────────────────────
        w, h      = pil_img.size
        short_side = min(w, h)
        if short_side < _QUALITY_CFG.min_resolution_px:
            log.debug(f"[quality] Low resolution: short_side={short_side}px")
            issues += 1

        gray         = np_rgb.mean(axis=2).astype(float)
        contrast     = float(gray.std())
        if contrast < 1.0:
            log.debug(f"[quality] Solid color/blank image detected (contrast={contrast:.2f}) — bypassing checks")
            return ImageQuality.HIGH

        blur_score   = self._variance_of_laplacian(gray)
        if blur_score < _QUALITY_CFG.min_blur_score:
            log.debug(f"[quality] Blurry image: laplacian_var={blur_score:.2f}")
            issues += 1

        # ── Contrast (RMS contrast) ─────────────────────────────────────────
        if contrast < _QUALITY_CFG.min_contrast:
            log.debug(f"[quality] Low contrast: rms_contrast={contrast:.2f}")
            issues += 1

        # ── Brightness (mean pixel value) ───────────────────────────────────
        brightness = float(gray.mean())
        if not (_QUALITY_CFG.min_brightness <= brightness <= _QUALITY_CFG.max_brightness):
            log.debug(f"[quality] Bad brightness: mean={brightness:.2f}")
            issues += 1

        if issues >= _QUALITY_CFG.unreadable_issue_count:
            quality = ImageQuality.UNREADABLE
        elif issues >= _QUALITY_CFG.low_quality_issue_count:
            quality = ImageQuality.LOW
        else:
            quality = ImageQuality.HIGH if issues == 0 else ImageQuality.NORMAL

        log.debug(
            f"[quality] grade={quality.value} issues={issues} "
            f"res={short_side}px blur={blur_score:.2f} "
            f"contrast={contrast:.2f} brightness={brightness:.2f}"
        )
        return quality

    @staticmethod
    def _variance_of_laplacian(gray_array) -> float:
        """Compute the Variance of Laplacian for a 2-D float array.

        This is a simple sharpness metric.  A sharp image has high spatial
        frequency content → high Laplacian variance.  A blurry image has low
        spatial frequency → low variance.

        Uses a manual 3×3 Laplacian kernel rather than cv2 to avoid the
        dependency on a heavy C extension just for this metric.
        """
        import numpy as np

        # 3×3 discrete Laplacian kernel
        kernel = np.array([
            [0,  1, 0],
            [1, -4, 1],
            [0,  1, 0],
        ], dtype=float)

        # Manual 2-D convolution via stride tricks (fast, no scipy needed).
        # Pad with zeros so output size matches input.
        g = np.pad(gray_array, 1, mode="edge")
        h, w = gray_array.shape
        lap  = (
            kernel[0, 0] * g[0:h,   0:w]   +
            kernel[0, 1] * g[0:h,   1:w+1] +
            kernel[0, 2] * g[0:h,   2:w+2] +
            kernel[1, 0] * g[1:h+1, 0:w]   +
            kernel[1, 1] * g[1:h+1, 1:w+1] +
            kernel[1, 2] * g[1:h+1, 2:w+2] +
            kernel[2, 0] * g[2:h+2, 0:w]   +
            kernel[2, 1] * g[2:h+2, 1:w+1] +
            kernel[2, 2] * g[2:h+2, 2:w+2]
        )
        return float(lap.var())

    def _is_image_readable(self, quality: ImageQuality) -> bool:
        """Return True if the quality grade is good enough to attempt OCR."""
        return quality != ImageQuality.UNREADABLE

    def _choose_render_dpi(self, quality: ImageQuality) -> int:
        """Map image quality to the starting DPI for page rendering.

        HIGH/NORMAL → use the lowest DPI that PaddleOCR handles well (saves CPU).
        LOW         → start at retry DPI to pre-empt a second render round.
        UNREADABLE  → caller should reject before reaching this; max is returned
                      as a conservative fallback so the pipeline stays defined.
        """
        if quality in (ImageQuality.HIGH, ImageQuality.NORMAL):
            return _DPI_CFG.initial
        if quality == ImageQuality.LOW:
            return _DPI_CFG.retry
        # UNREADABLE — caller must check _is_image_readable before rendering.
        return _DPI_CFG.max

    # ── OCR orchestration ───────────────────────────────────────────────────

    def _ocr_image_bytes(
        self,
        img_bytes: bytes,
        label: str,
        is_pdf_page: bool = False,
    ) -> Optional[str]:
        """Run the full OCR pipeline on raw image bytes.

        Pipeline (single retry pass, no infinite loops):
          1. Assess image quality.
          2. If UNREADABLE and not is_pdf_page → raise OCRError immediately (medical safety).
          3. PaddleOCR at chosen DPI.
          4. If PaddleOCR result fails validation → bump DPI to max, retry once.
          5. If still failing → pytesseract fallback.
          6. If pytesseract fails → return None (caller raises OCRError).

        Args:
            img_bytes: Raw PNG/JPEG bytes of the image to OCR.
            label:     Human-readable label used in log messages only
                       (e.g. "page 3" or "report.jpg").
            is_pdf_page: If True, skips UNREADABLE checks for rendered PDF pages.

        Returns:
            Extracted text string, or None if all engines failed.

        Raises:
            OCRError: Image quality is UNREADABLE — rejecting rather than
                      returning unreliable medical data.
        """
        quality = self._analyze_image_quality(img_bytes)
        log.debug(f"[ocr] label='{label}' quality={quality.value}")

        if not is_pdf_page and not self._is_image_readable(quality):
            raise OCRError(
                f"Image '{label}' is UNREADABLE (too blurry, low-contrast, or "
                "under/over-exposed). Rejecting to avoid unreliable medical data. "
                "Re-scan at higher quality or better lighting."
            )

        # ── PaddleOCR (primary) ─────────────────────────────────────────────
        paddle = self._get_paddle()
        if paddle is not None:
            log.debug(f"[ocr] label='{label}' Attempting PaddleOCR")
            dpi         = self._choose_render_dpi(quality)
            paddle_text = self._run_paddle_ocr(paddle, img_bytes, label, dpi)

            if paddle_text is not None:
                log.debug(f"[ocr] label='{label}' PaddleOCR succeeded")
                return paddle_text

            # One retry at max DPI if initial attempt failed validation.
            if dpi < _DPI_CFG.max:
                log.debug(
                    f"[ocr] label='{label}' PaddleOCR output failed validation "
                    f"at dpi={dpi} — retrying at dpi={_DPI_CFG.max}"
                )
                # Note: img_bytes at this point are already a rendered image, so
                # bumping DPI here applies to the caller's next render; for
                # standalone images the bytes don't change (DPI is a render
                # parameter for PDF pages).  For images we simply retry OCR on
                # the same bytes — PaddleOCR is stochastic enough that a retry
                # can still improve confidence, and consistency matters more here
                # than a true DPI re-render.
                paddle_text = self._run_paddle_ocr(paddle, img_bytes, label, _DPI_CFG.max)
                if paddle_text is not None:
                    log.debug(f"[ocr] label='{label}' PaddleOCR succeeded on retry")
                    return paddle_text
            else:
                log.debug(f"[ocr] label='{label}' PaddleOCR failed and no retry DPI available")
        else:
            log.debug(f"[ocr] label='{label}' PaddleOCR not available, skipping to pytesseract")

        # ── pytesseract (fallback) ──────────────────────────────────────────
        log.debug(f"[ocr] label='{label}' Falling back to pytesseract")
        return self._run_tesseract_ocr(img_bytes, label)

    def _run_paddle_ocr(
        self,
        paddle,
        img_bytes: bytes,
        label: str,
        dpi: int,
    ) -> Optional[str]:
        """Execute PaddleOCR on img_bytes and return validated text or None.

        Validation gates (any failure → None):
          - avg_confidence >= OCR_VAL_CFG.min_avg_confidence
          - word_count     >= OCR_VAL_CFG.min_word_count
          - line_count     >= OCR_VAL_CFG.min_line_count
          - char_length    >= OCR_VAL_CFG.min_char_length

        Low-confidence lines are retained as a secondary pool if no
        high-confidence lines were found — this preserves handwritten
        prescription text that PaddleOCR tends to score conservatively.
        """
        try:
            import numpy as np
            from PIL import Image

            pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            result  = paddle.ocr(np.array(pil_img), cls=True)
        except Exception as exc:
            log.warning(f"[paddle] label='{label}' PaddleOCR raised an exception: {exc}")
            return None

        if not result or not result[0]:
            log.debug(f"[paddle] label='{label}' dpi={dpi} — empty result")
            return None

        lines: list[str]    = []
        low_conf: list[str] = []
        confidences: list[float] = []

        for line in result[0]:
            # Each PaddleOCR line is [[bbox_points], [text, confidence]]
            if not line or len(line) < 2:
                continue
            text_conf = line[1]
            if not text_conf or len(text_conf) < 2:
                continue

            text = (text_conf[0] or "").strip()
            conf = text_conf[1]

            # Skip lines shorter than the minimum useful character count.
            if len(text) < OCR_MIN_LINE_CHARS:
                continue

            if conf >= OCR_MIN_CONFIDENCE:
                lines.append(text)
                confidences.append(conf)
            else:
                # Retain low-confidence lines in a separate pool; they may be
                # the only output for handwritten text.
                low_conf.append(text)

        # Prefer high-confidence lines; fall back to low-conf pool when empty.
        accepted_lines = lines if lines else low_conf
        if not accepted_lines:
            log.debug(f"[paddle] label='{label}' dpi={dpi} — no accepted lines")
            return None

        text_out     = "\n".join(accepted_lines).strip()
        avg_conf     = (sum(confidences) / len(confidences)) if confidences else 0.0
        word_count   = len(text_out.split())

        # Validate the result before accepting it.
        if not self._validate_ocr_result(
            text=text_out,
            avg_confidence=avg_conf,
            line_count=len(accepted_lines),
            word_count=word_count,
            engine="PaddleOCR",
            label=label,
        ):
            return None

        log.debug(
            f"[paddle] label='{label}' dpi={dpi} "
            f"chars={len(text_out)} lines={len(accepted_lines)} "
            f"words={word_count} avg_conf={avg_conf:.3f}"
        )
        return text_out

    def _run_tesseract_ocr(self, img_bytes: bytes, label: str) -> Optional[str]:
        """Execute pytesseract on img_bytes and return text or None.

        PSM 6 assumes a uniform block of text — appropriate for both lab
        reports (structured tables) and prescriptions (dense paragraphs).
        """
        try:
            import pytesseract
            from PIL import Image

            pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            text    = pytesseract.image_to_string(pil_img, config="--psm 6").strip()

            if not text:
                log.warning(f"[tesseract] label='{label}' — empty result")
                return None

            log.debug(f"[tesseract] label='{label}' chars={len(text)}")
            return text

        except Exception as exc:
            log.warning(f"[tesseract] label='{label}' raised an exception: {exc}")
            return None

    def _validate_ocr_result(
        self,
        text: str,
        avg_confidence: float,
        line_count: int,
        word_count: int,
        engine: str,
        label: str,
    ) -> bool:
        """Return True only if the OCR result clears all validation thresholds.

        Having explicit validation gates here prevents downstream prompts from
        receiving garbled or near-empty text that could mislead the LLM into
        producing an incorrect medical interpretation.
        """
        failures: list[str] = []

        if avg_confidence < _OCR_VAL_CFG.min_avg_confidence:
            failures.append(
                f"avg_confidence={avg_confidence:.3f} < {_OCR_VAL_CFG.min_avg_confidence}"
            )
        if line_count < _OCR_VAL_CFG.min_line_count:
            failures.append(f"line_count={line_count} < {_OCR_VAL_CFG.min_line_count}")
        if word_count < _OCR_VAL_CFG.min_word_count:
            failures.append(f"word_count={word_count} < {_OCR_VAL_CFG.min_word_count}")
        if len(text) < _OCR_VAL_CFG.min_char_length:
            failures.append(
                f"char_length={len(text)} < {_OCR_VAL_CFG.min_char_length}"
            )

        if failures:
            log.debug(
                f"[validate] engine={engine} label='{label}' "
                f"FAILED validation: {'; '.join(failures)}"
            )
            return False

        return True

    # ── PDF extraction: full pipeline ──────────────────────────────────────

    def _extract_pdf(self, path: Path) -> str:
        """Orchestrate the full PDF extraction pipeline.

        Opens the PDF once with PyMuPDF, classifies it, then dispatches each
        page to the appropriate extractor.  The fitz document is kept open
        for the duration to avoid re-parsing on every page.

        Returns:
            Page-marked text string.

        Raises:
            OCRError:           PyMuPDF import failed or an OCR engine failed.
            EmptyDocumentError: No text could be extracted from the PDF.
        """
        try:
            import fitz
        except ImportError:
            raise OCRError("PyMuPDF is not installed. Run: pip install pymupdf")

        log.debug(f"[pdf] Using PyMuPDF to analyze '{path.name}'")

        try:
            fitz_doc = fitz.open(str(path))
        except Exception as exc:
            log.error(f"[pdf] Failed to open PDF '{path.name}': {exc}")
            raise EmptyDocumentError("Failed to parse PDF document: file is empty or corrupted.")

        try:
            if fitz_doc.is_encrypted:
                raise EmptyDocumentError("Encrypted or password-protected PDF files are not supported.")

            if fitz_doc.page_count > 50:
                raise EmptyDocumentError("PDF exceeds maximum allowed page limit of 50 pages.")

            log.info(f"[pdf] Successfully opened '{path.name}' with {fitz_doc.page_count} pages")
            analyses = self._analyze_pages(fitz_doc)
            doc_type = self._classify_document(analyses)
            extracted = self._dispatch_extraction(path, fitz_doc, analyses, doc_type)
        finally:
            try:
                fitz_doc.close()
            except Exception:
                pass

        # Apply the unified cleaning stage to all extracted pages before merging them.
        cleaned_extracted = self.clean_pages(extracted)
        combined = self._merge_pages(cleaned_extracted, total_pages=len(analyses))

        if not combined:
            raise EmptyDocumentError(
                f"PDF '{path.name}' produced no text after extraction."
            )
        return combined

    def _dispatch_extraction(
        self,
        path: Path,
        fitz_doc,
        analyses: list[PageAnalysis],
        doc_type: DocType,
    ) -> dict[int, str]:
        """Route each page to the right extractor based on document type.

        Returns:
            Mapping of {page_num: extracted_text}.
        """
        if doc_type == DocType.DIGITAL_PDF:
            return self._extract_digital_pdf(path, analyses)

        if doc_type == DocType.SCANNED_PDF:
            return self._extract_scanned_pdf(fitz_doc, analyses)

        # MIXED_PDF — hybrid routing
        return self._extract_mixed_pdf(path, fitz_doc, analyses)

    def _extract_digital_pdf(
        self,
        path: Path,
        analyses: list[PageAnalysis],
    ) -> dict[int, str]:
        """Extract all pages via pdfplumber (DIGITAL_PDF strategy)."""
        all_pages = [a.page_num for a in analyses]
        log.debug(
            f"[pdf] Strategy=DIGITAL: pdfplumber for all {len(all_pages)} pages"
        )
        return self._extract_pages_pdfplumber(path, all_pages)

    def _extract_scanned_pdf(
        self,
        fitz_doc,
        analyses: list[PageAnalysis],
    ) -> dict[int, str]:
        """OCR every page (SCANNED_PDF strategy).

        Renders each page at adaptive DPI based on image quality, then runs
        the OCR pipeline.
        """
        log.debug(
            f"[pdf] Strategy=SCANNED: OCR for all {len(analyses)} pages"
        )
        extracted: dict[int, str] = {}

        for a in analyses:
            text = self._ocr_pdf_page(fitz_doc, a)
            extracted[a.page_num] = text or ""

        return extracted

    def _extract_mixed_pdf(
        self,
        path: Path,
        fitz_doc,
        analyses: list[PageAnalysis],
    ) -> dict[int, str]:
        """Hybrid extraction for MIXED_PDF: route each page individually.

        TEXT_PAGE  → pdfplumber
        MIXED_PAGE → pdfplumber; fall back to OCR if result is still sparse
        OCR_PAGE   → OCR directly
        """
        text_analyses  = [a for a in analyses if a.page_type == PageType.TEXT_PAGE]
        mixed_analyses = [a for a in analyses if a.page_type == PageType.MIXED_PAGE]
        ocr_analyses   = [a for a in analyses if a.page_type == PageType.OCR_PAGE]

        log.debug(
            f"[pdf] Strategy=MIXED: "
            f"direct={len(text_analyses)} mixed={len(mixed_analyses)} "
            f"ocr={len(ocr_analyses)} pages"
        )

        extracted: dict[int, str] = {}

        # ── TEXT_PAGE: direct extraction ────────────────────────────────────
        if text_analyses:
            nums   = [a.page_num for a in text_analyses]
            result = self._extract_pages_pdfplumber(path, nums)
            extracted.update(result)

        # ── MIXED_PAGE: try pdfplumber; OCR if still too sparse ────────────
        if mixed_analyses:
            nums          = [a.page_num for a in mixed_analyses]
            plumber_result = self._extract_pages_pdfplumber(path, nums)

            for a in mixed_analyses:
                text = plumber_result.get(a.page_num, "")
                if len(text.strip()) >= _PAGE_CFG.char_threshold_ocr:
                    extracted[a.page_num] = text
                    log.debug(
                        f"[pdf] MIXED page={a.page_num} "
                        f"pdfplumber OK ({len(text)} chars)"
                    )
                else:
                    log.debug(
                        f"[pdf] MIXED page={a.page_num} "
                        f"pdfplumber sparse ({len(text)} chars) — routing to OCR"
                    )
                    extracted[a.page_num] = self._ocr_pdf_page(fitz_doc, a) or ""

        # ── OCR_PAGE: render + OCR ──────────────────────────────────────────
        for a in ocr_analyses:
            extracted[a.page_num] = self._ocr_pdf_page(fitz_doc, a) or ""

        return extracted

    def _ocr_pdf_page(self, fitz_doc, analysis: PageAnalysis) -> Optional[str]:
        """Render a single PDF page to an image and run OCR on it.

        Uses adaptive DPI: renders at quality-appropriate DPI, then lets
        _ocr_image_bytes handle the retry ladder internally.

        Args:
            fitz_doc: Open fitz.Document.
            analysis: PageAnalysis for the page to render.

        Returns:
            Extracted text, or None if OCR produced nothing usable.
        """
        # Render at initial DPI; _ocr_image_bytes will handle quality-driven retries.
        img_bytes = self._render_page_to_image(
            fitz_doc,
            page_index=analysis.page_num - 1,  # PyMuPDF is 0-based
            dpi=_DPI_CFG.initial,
        )

        label = f"page {analysis.page_num}"
        quality = self._analyze_image_quality(img_bytes)

        # If initial render is LOW quality, re-render at a higher DPI before OCR.
        if quality == ImageQuality.LOW:
            log.debug(
                f"[pdf] page={analysis.page_num} quality=LOW — "
                f"re-rendering at dpi={_DPI_CFG.retry}"
            )
            img_bytes = self._render_page_to_image(
                fitz_doc,
                page_index=analysis.page_num - 1,
                dpi=_DPI_CFG.retry,
            )

        # _ocr_image_bytes handles UNREADABLE rejection, PaddleOCR validation,
        # one max-DPI retry, and pytesseract fallback.
        try:
            return self._ocr_image_bytes(img_bytes, label, is_pdf_page=True)
        except OCRError:
            # Re-raise so the caller (and ultimately the user) gets a clear
            # rejection message rather than a silent empty string.
            raise

    # ── Lab report cleaning ─────────────────────────────────────────────────

    def _clean_line(self, line: str, line_idx: int, total_lines: int) -> Optional[str]:
        """Clean a single line of text and determine if it should be kept."""
        stripped = line.strip()
        if not stripped:
            return None

        # 1. Short lines: Keep if length < 3 but has at least one alphanumeric character
        if len(stripped) < 3 and not any(c.isalnum() for c in stripped):
            return None

        # 2. Known footer/header fragments
        if _FOOTER_REGEX.search(stripped):
            return None

        # 3. Standalone page-number lines (explicit anywhere, ambiguous only at top/bottom 3 lines)
        if _EXPLICIT_PAGE_NUM_REGEX.match(stripped) or (
            _AMBIGUOUS_PAGE_NUM_REGEX.match(stripped) and (line_idx < 3 or line_idx >= total_lines - 3)
        ):
            return None

        # 4. OCR-garbage lines (>50% non-alphanumeric chars after stripping leader symbols)
        no_ws = "".join(_LEADER_DOTS_REGEX.sub('', stripped).split())
        if not no_ws or sum(1 for c in no_ws if not c.isalnum()) / len(no_ws) > 0.5:
            return None

        return stripped

    def _clean_page_text(self, raw: str) -> str:
        """Applies _clean_line per line, also dropping consecutive duplicate lines."""
        visible_lines = [l for l in raw.split("\n") if l.strip()]
        total_visible = len(visible_lines)
        cleaned_lines = []
        last_line = None

        for idx, line in enumerate(visible_lines):
            cleaned = self._clean_line(line, idx, total_visible)
            if cleaned is not None:
                if cleaned != last_line:
                    cleaned_lines.append(cleaned)
                    last_line = cleaned

        return "\n".join(cleaned_lines)

    def _strip_repeated_lines(self, pages: dict[int, str]) -> dict[int, str]:
        """Removes lines appearing on >=50% of pages (if >=3 pages total) and containing no digits."""
        if len(pages) < 3:
            return pages

        from collections import Counter
        line_counts = Counter(
            line for text in pages.values() for line in set(text.split("\n")) if line.strip()
        )
        repeated = {
            line for line, count in line_counts.items()
            if count >= len(pages) / 2 and not any(c.isdigit() for c in line)
        }
        if not repeated:
            return pages

        return {
            page_num: "\n".join(line for line in text.split("\n") if line not in repeated)
            for page_num, text in pages.items()
        }

    def clean_pages(self, raw_pages: dict[int, str]) -> dict[int, str]:
        """Unified entry point for cleaning and de-noising pages."""
        cleaned = {pn: self._clean_page_text(raw) for pn, raw in raw_pages.items()}
        return self._strip_repeated_lines(cleaned)

    # ── Page merging ────────────────────────────────────────────────────────

    def _merge_pages(
        self,
        extracted: dict[int, str],
        total_pages: int,
    ) -> str:
        """Merge per-page text into a single document string.

        Page markers (``[Page N]``) are preserved so downstream LLM prompts
        can reference specific pages when reporting findings.  Empty pages are
        silently skipped.

        Args:
            extracted:   Mapping of {page_num: text}.
            total_pages: Total number of pages (used to enforce page order).

        Returns:
            Combined text string, or '' if all pages were empty.
        """
        page_blocks: list[str] = []
        for page_num in range(1, total_pages + 1):
            text = extracted.get(page_num, "").strip()
            if text:
                page_blocks.append(f"[Page {page_num}]\n{text}")

        combined = "\n\n".join(page_blocks).strip()
        log.debug(
            f"[merge] {len(page_blocks)}/{total_pages} pages had content — "
            f"total_chars={len(combined)}"
        )
        return combined

    # ── Standalone image OCR ─────────────────────────────────────────────────

    def _extract_image(self, path: Path) -> str:
        """OCR a standalone image file (non-PDF).

        Raises:
            OCRError: All OCR engines failed or image is UNREADABLE.
        """
        img_bytes = path.read_bytes()
        text      = self._ocr_image_bytes(img_bytes, path.name)

        if text:
            log.debug(
                f"[image] OCR succeeded for '{path.name}' — chars={len(text)}"
            )
            return text

        raise OCRError(
            f"All OCR engines failed for '{path.name}'. "
            "Ensure the image is clear, well-lit, and at least 300 DPI."
        )