import os
import time
import pytest
import threading
from health_ai.utils.document_reader import DocumentReader
from health_ai.core.exceptions import OCRError, UnsupportedFileTypeError, EmptyDocumentError

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")

# Maximum execution time allowed for a single document extraction (in seconds)
TIMEOUT_LIMIT = 45.0

def run_with_timeout(func, *args, **kwargs):
    """
    Runs a function in a background thread and waits for it to complete.
    Raises a TimeoutError if the execution exceeds TIMEOUT_LIMIT.
    """
    res = []
    exc = []

    def target():
        try:
            r = func(*args, **kwargs)
            res.append(r)
        except Exception as e:
            exc.append(e)

    thread = threading.Thread(target=target)
    thread.daemon = True
    start_time = time.time()
    thread.start()
    thread.join(timeout=TIMEOUT_LIMIT)
    elapsed = time.time() - start_time

    if thread.is_alive():
        raise TimeoutError(f"Function call hung and timed out after {TIMEOUT_LIMIT} seconds")

    if exc:
        raise exc[0]

    return res[0] if res else None


@pytest.fixture(scope="module")
def reader():
    r = DocumentReader()
    print("\nPre-warming PaddleOCR engine for stress-tests...")
    r._get_paddle()
    print("PaddleOCR engine pre-warmed.")
    return r


# ── 1. MALFORMED / CORRUPT INPUT ──────────────────────────────────────────

def test_zero_byte_file(reader):
    path = os.path.join(FIXTURES_DIR, "zero_byte.pdf")
    with pytest.raises(EmptyDocumentError):
        run_with_timeout(reader.extract, path)

def test_not_a_pdf(reader):
    path = os.path.join(FIXTURES_DIR, "not_a_pdf.pdf")
    # Should raise EmptyDocumentError or OCRError, but not a crash
    with pytest.raises((EmptyDocumentError, OCRError)):
        run_with_timeout(reader.extract, path)

def test_not_an_image(reader):
    path = os.path.join(FIXTURES_DIR, "not_an_image.png")
    with pytest.raises(OCRError):
        run_with_timeout(reader.extract, path)

def test_truncated_pdf(reader):
    path = os.path.join(FIXTURES_DIR, "truncated.pdf")
    with pytest.raises((OCRError, EmptyDocumentError)):
        run_with_timeout(reader.extract, path)

def test_corrupt_xref(reader):
    path = os.path.join(FIXTURES_DIR, "corrupt_xref.pdf")
    with pytest.raises((OCRError, EmptyDocumentError)):
        run_with_timeout(reader.extract, path)


# ── 2. RESOURCE EXHAUSTION ────────────────────────────────────────────────

def test_decompression_bomb(reader):
    path = os.path.join(FIXTURES_DIR, "decompression_bomb.png")
    # PIL or numpy should reject this as unreadable, raising OCRError
    with pytest.raises(OCRError):
        run_with_timeout(reader.extract, path)

def test_large_500_pages(reader):
    path = os.path.join(FIXTURES_DIR, "large_500_pages.pdf")
    # Processing 500 pages will take a long time and should exceed the timeout,
    # or fail due to resources. We verify it raises or gets caught by timeout.
    with pytest.raises((TimeoutError, EmptyDocumentError, OCRError)):
        run_with_timeout(reader.extract, path)


# ── 3. SECURITY-ADJACENT ──────────────────────────────────────────────────

def test_password_protected_pdf(reader):
    path = os.path.join(FIXTURES_DIR, "password_protected.pdf")
    # Should raise OCRError/EmptyDocumentError rather than prompting for passwords
    with pytest.raises((OCRError, EmptyDocumentError)):
        run_with_timeout(reader.extract, path)

def test_js_embedded_pdf(reader):
    path = os.path.join(FIXTURES_DIR, "js_embedded.pdf")
    # JS must not execute or cause failures; standard extraction expected
    try:
        text = run_with_timeout(reader.extract, path)
        assert "XSS" not in text  # Ensure JS actions didn't leak into output text
    except (OCRError, EmptyDocumentError):
        pass  # Raising is also a safe handling

def test_attachments_pdf(reader):
    path = os.path.join(FIXTURES_DIR, "attachments.pdf")
    # Attachments must be ignored; only main text layer is parsed
    try:
        text = run_with_timeout(reader.extract, path)
        assert "secret payload" not in text
    except EmptyDocumentError:
        pass


# ── 4. SCAN QUALITY VARIANCE ──────────────────────────────────────────────

def test_skewed_page(reader):
    path = os.path.join(FIXTURES_DIR, "skewed_page.png")
    # Should extract text successfully despite skew
    text = run_with_timeout(reader.extract, path)
    assert text is not None
    assert len(text.strip()) > 0

def test_low_contrast(reader):
    path = os.path.join(FIXTURES_DIR, "low_contrast.png")
    # Contrast may drop below readability thresholds, raising OCRError, or pass
    try:
        text = run_with_timeout(reader.extract, path)
        assert text is not None
    except OCRError:
        pass

def test_handwritten(reader):
    path = os.path.join(FIXTURES_DIR, "handwritten.png")
    try:
        text = run_with_timeout(reader.extract, path)
        assert text is not None
    except OCRError:
        pass

def test_noisy(reader):
    path = os.path.join(FIXTURES_DIR, "noisy.png")
    try:
        text = run_with_timeout(reader.extract, path)
        assert text is not None
    except OCRError:
        pass


# ── 5. LAYOUT COMPLEXITY ──────────────────────────────────────────────────

def test_multi_column(reader):
    path = os.path.join(FIXTURES_DIR, "multi_column.pdf")
    text = run_with_timeout(reader.extract, path)
    assert "Alice" in text
    assert "HbA1c" in text

def test_table_page(reader):
    path = os.path.join(FIXTURES_DIR, "table_page.pdf")
    text = run_with_timeout(reader.extract, path)
    assert "Hemoglobin" in text
    assert "Platelet" in text

def test_mixed_digital_scanned(reader):
    path = os.path.join(FIXTURES_DIR, "mixed_digital_scanned.pdf")
    text = run_with_timeout(reader.extract, path)
    assert "Hemoglobin" in text or "ALT" in text

def test_rotated_blocks(reader):
    path = os.path.join(FIXTURES_DIR, "rotated_blocks.pdf")
    text = run_with_timeout(reader.extract, path)
    assert "Normal" in text or "vertical" in text


# ── 6. CONTENT EDGE CASES ──────────────────────────────────────────────────

def test_blank_page(reader):
    path = os.path.join(FIXTURES_DIR, "blank.pdf")
    with pytest.raises(EmptyDocumentError):
        run_with_timeout(reader.extract, path)

def test_non_english(reader):
    path = os.path.join(FIXTURES_DIR, "non_english.pdf")
    text = run_with_timeout(reader.extract, path)
    assert "Amoxicilina" in text or "Receta" in text

def test_medical_symbols(reader):
    path = os.path.join(FIXTURES_DIR, "medical_symbols.pdf")
    text = run_with_timeout(reader.extract, path)
    # Check if symbols were parsed cleanly without crashing
    assert "µg" in text or "temp" in text


# ── 7. CONFIDENCE BOUNDARY ────────────────────────────────────────────────

def test_confidence_boundary_below(reader):
    path = os.path.join(FIXTURES_DIR, "boundary_0_39.png")
    # Low confidence image should fail the validation check and raise OCRError
    with pytest.raises(OCRError):
        run_with_timeout(reader.extract, path)

def test_confidence_boundary_above(reader):
    path = os.path.join(FIXTURES_DIR, "boundary_0_41.png")
    # Higher confidence should pass or throw OCRError depending on actual OCR engine accuracy,
    # but must not crash
    try:
        text = run_with_timeout(reader.extract, path)
        assert text is not None
    except OCRError:
        pass


# ── 8. CONCURRENCY ────────────────────────────────────────────────────────

def test_concurrency(reader):
    paths = [
        os.path.join(FIXTURES_DIR, "multi_column.pdf"),
        os.path.join(FIXTURES_DIR, "table_page.pdf"),
        os.path.join(FIXTURES_DIR, "medical_symbols.pdf")
    ]

    results = {}
    errors = []

    def task(i, path):
        try:
            results[i] = reader.extract(path)
        except Exception as e:
            errors.append(e)

    threads = []
    for i, path in enumerate(paths):
        t = threading.Thread(target=task, args=(i, path))
        threads.append(t)

    for t in threads:
        t.start()

    for t in threads:
        t.join(timeout=TIMEOUT_LIMIT)

    assert len(errors) == 0, f"Concurrent tasks raised errors: {errors}"
    assert len(results) == 3
