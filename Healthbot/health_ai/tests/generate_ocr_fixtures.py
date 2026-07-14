import os
import io
import fitz
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")
os.makedirs(FIXTURES_DIR, exist_ok=True)

def generate_all():
    print(f"Generating fixtures in: {FIXTURES_DIR}")

    # 1. Zero-byte file
    with open(os.path.join(FIXTURES_DIR, "zero_byte.pdf"), "wb") as f:
        f.write(b"")

    # 2. Non-PDF file renamed to .pdf
    with open(os.path.join(FIXTURES_DIR, "not_a_pdf.pdf"), "w", encoding="utf-8") as f:
        f.write("This is a plain text file, not a PDF.")

    # 3. Non-image file renamed to .png
    with open(os.path.join(FIXTURES_DIR, "not_an_image.png"), "w", encoding="utf-8") as f:
        f.write("This is a plain text file, not a PNG image.")

    # 4. Truncated PDF
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), "This is a truncated PDF file for testing.")
    pdf_bytes = doc.write()
    doc.close()
    with open(os.path.join(FIXTURES_DIR, "truncated.pdf"), "wb") as f:
        f.write(pdf_bytes[:len(pdf_bytes) // 2])  # slice in half

    # 5. Corrupt Xref table
    # Just write some corrupted PDF-like headers and trailing garbage
    with open(os.path.join(FIXTURES_DIR, "corrupt_xref.pdf"), "wb") as f:
        f.write(b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n2 0 obj\n<<\n/Type /Pages\n/Count 0\n>>\nendobj\nxref\n0 3\n0000000000 65535 f\n9999999999 00000 n\n9999999999 00000 n\ntrailer\n<<\n/Size 3\n/Root 1 0 R\n>>\nstartxref\n123456\n%%EOF")

    # 6. Decompression bomb image
    # A small compressed file with 8000x8000 pixels of solid white
    img = Image.new("RGB", (8000, 8000), "white")
    img.save(os.path.join(FIXTURES_DIR, "decompression_bomb.png"), "PNG", compress_level=9)

    # 7. Large 500 pages PDF
    doc = fitz.open()
    for i in range(1, 501):
        p = doc.new_page()
        p.insert_text((100, 100), f"Page {i} content. This is a large PDF stress test.")
    doc.save(os.path.join(FIXTURES_DIR, "large_500_pages.pdf"))
    doc.close()

    # 8. Password protected PDF
    doc = fitz.open()
    p = doc.new_page()
    p.insert_text((50, 50), "This PDF is encrypted with user password 'secret'.")
    # PyMuPDF 1.22+ save encrypt signature:
    doc.save(os.path.join(FIXTURES_DIR, "password_protected.pdf"), encryption=fitz.PDF_ENCRYPT_AES_256, user_pw="secret", owner_pw="owner")
    doc.close()

    # 9. PDF with Embedded JS
    with open(os.path.join(FIXTURES_DIR, "js_embedded.pdf"), "wb") as f:
        f.write(b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n/OpenAction << /Type /Action /S /JavaScript /JS (app.alert('XSS');) >>\n>>\nendobj\n2 0 obj\n<<\n/Type /Pages\n/Count 1\n/Kids [3 0 R]\n>>\nendobj\n3 0 obj\n<<\n/Type /Page\n/Parent 2 0 R\n/MediaBox [0 0 612 792]\n/Resources << >>\n>>\nendobj\nxref\n0 4\n0000000000 65535 f\n0000000009 00000 n\n0000000115 00000 n\n0000000178 00000 n\ntrailer\n<<\n/Size 4\n/Root 1 0 R\n>>\nstartxref\n268\n%%EOF")

    # 10. PDF with attachments
    doc = fitz.open()
    p = doc.new_page()
    p.insert_text((50, 50), "This PDF contains a file attachment.")
    doc.embfile_add(name="attached.txt", buffer_=b"This is an embedded file payload.", filename="attached.txt")
    doc.save(os.path.join(FIXTURES_DIR, "attachments.pdf"))
    doc.close()

    # 11. Skewed / Rotated page image
    # Generate an image with text, then rotate it
    img_txt = Image.new("RGB", (800, 400), "white")
    draw = ImageDraw.Draw(img_txt)
    draw.text((10, 150), "Patient Name: John Doe\nRx: Amoxicillin 500mg daily", fill="black")
    rotated_img = img_txt.rotate(15, expand=True, fillcolor="white")
    rotated_img.save(os.path.join(FIXTURES_DIR, "skewed_page.png"))

    # 12. Low contrast scan image
    # Grey text on slightly lighter grey background
    img_contrast = Image.new("RGB", (800, 400), "#E0E0E0")
    draw = ImageDraw.Draw(img_contrast)
    draw.text((10, 150), "Patient Name: Jane Smith\nLab: ALT 45 U/L (Borderline)", fill="#909090")
    img_contrast.save(os.path.join(FIXTURES_DIR, "low_contrast.png"))

    # 13. Handwritten text simulation
    img_hand = Image.new("RGB", (800, 400), "white")
    draw = ImageDraw.Draw(img_hand)
    # Simulate sloppy lines representing handwritten medical shorthand
    draw.line([(20, 100), (300, 100)], fill="black", width=2)
    draw.text((30, 80), "Rx: Metformin 1000mg BID", fill="blue")
    img_hand.save(os.path.join(FIXTURES_DIR, "handwritten.png"))

    # 14. Noisy scan image
    img_noisy = Image.new("RGB", (800, 400), "white")
    draw = ImageDraw.Draw(img_noisy)
    draw.text((50, 150), "Blood Report: WBC 11.5 x10^9/L (Elevated)", fill="black")
    # Add salt and pepper noise
    arr = np.array(img_noisy)
    noise = np.random.randint(0, 255, arr.shape, dtype=np.uint8)
    noisy_arr = np.where(np.random.rand(*arr.shape[:2])[:,:,None] > 0.9, noise, arr)
    img_noisy_noisy = Image.fromarray(noisy_arr)
    img_noisy_noisy.save(os.path.join(FIXTURES_DIR, "noisy.png"))

    # 15. Multi-column PDF
    doc = fitz.open()
    p = doc.new_page()
    # Left column
    p.insert_text((50, 100), "COLUMN ONE\nPatient: Alice Cooper\nAge: 45\nGender: Female")
    # Right column
    p.insert_text((300, 100), "COLUMN TWO\nLab Results:\nHbA1c: 6.2 %\nTSH: 2.1 uIU/mL")
    doc.save(os.path.join(FIXTURES_DIR, "multi_column.pdf"))
    doc.close()

    # 16. Tables PDF
    doc = fitz.open()
    p = doc.new_page()
    p.insert_text((50, 50), "Complete Blood Count Table")
    # Draw simple table border lines
    p.draw_line((50, 80), (500, 80))
    p.draw_line((50, 120), (500, 120))
    p.insert_text((60, 95), "Test Name      | Value | Reference Range")
    p.insert_text((60, 140), "Hemoglobin     | 14.2  | 12.0 - 16.0 g/dL")
    p.insert_text((60, 160), "Platelet Count | 250   | 150 - 450 x10^9/L")
    doc.save(os.path.join(FIXTURES_DIR, "table_page.pdf"))
    doc.close()

    # 17. Mixed digital and scanned page PDF
    doc = fitz.open()
    # Page 1: Digital text
    p1 = doc.new_page()
    p1.insert_text((50, 100), "This is a digital text page. Hemoglobin is 13.5 g/dL.")
    # Page 2: Scanned page image inserted as image block
    p2 = doc.new_page()
    # Render the low_contrast.png into bytes to embed it as an image block
    img_buf = io.BytesIO()
    img_contrast.save(img_buf, format="PNG")
    p2.insert_image(p2.rect, stream=img_buf.getvalue())
    doc.save(os.path.join(FIXTURES_DIR, "mixed_digital_scanned.pdf"))
    doc.close()

    # 18. Rotated text blocks PDF
    doc = fitz.open()
    p = doc.new_page()
    p.insert_text((50, 100), "Normal horizontal text block.")
    # Insert text rotated by 90 degrees
    # fitz insert_text accepts rotate parameter (0, 90, 180, 270)
    p.insert_text((300, 300), "Rotated vertical text block.", rotate=90)
    doc.save(os.path.join(FIXTURES_DIR, "rotated_blocks.pdf"))
    doc.close()

    # 19. Blank page
    doc = fitz.open()
    doc.new_page()
    doc.save(os.path.join(FIXTURES_DIR, "blank.pdf"))
    doc.close()

    # 20. Non-English text
    doc = fitz.open()
    p = doc.new_page()
    p.insert_text((50, 100), "Receta Médica:\nTomar Amoxicilina 500mg cada 8 horas por 7 días.")
    doc.save(os.path.join(FIXTURES_DIR, "non_english.pdf"))
    doc.close()

    # 21. Medical symbols
    doc = fitz.open()
    p = doc.new_page()
    p.insert_text((50, 100), "Dosage: 250 µg ± 5% at temp 37 °C.")
    doc.save(os.path.join(FIXTURES_DIR, "medical_symbols.pdf"))
    doc.close()

    # 22. Boundary low contrast 0.39 (intended to fail OCR_MIN_CONFIDENCE gate)
    img_b39 = Image.new("RGB", (600, 300), "#FFFFFF")
    draw = ImageDraw.Draw(img_b39)
    # extremely light gray text, highly blurred to reduce OCR confidence below 0.40
    draw.text((10, 100), "HbA1c 9.5 % (Critical)", fill="#EAEAEA")
    img_b39_blurred = img_b39.filter(ImageFilter.GaussianBlur(2))
    img_b39_blurred.save(os.path.join(FIXTURES_DIR, "boundary_0_39.png"))

    # 23. Boundary 0.41 (intended to pass OCR_MIN_CONFIDENCE gate)
    img_b41 = Image.new("RGB", (600, 300), "#FFFFFF")
    draw = ImageDraw.Draw(img_b41)
    # slightly darker gray text, mildly blurred to keep OCR confidence just above 0.40
    draw.text((10, 100), "HbA1c 9.5 % (Critical)", fill="#C0C0C0")
    img_b41_blurred = img_b41.filter(ImageFilter.GaussianBlur(0.8))
    img_b41_blurred.save(os.path.join(FIXTURES_DIR, "boundary_0_41.png"))

    print("All fixtures generated successfully.")

if __name__ == "__main__":
    generate_all()
