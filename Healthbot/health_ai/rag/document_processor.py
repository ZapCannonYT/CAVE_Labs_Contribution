"""
document_processor.py — Heuristic-based patient data formatter and hybrid chunker.
Extracts structured medical measurements and schedules without LLM invocation,
and yields both a high-density structured summary chunk and standard raw chunks.
"""

import re
from typing import List
from health_ai.rag.chunker import Chunk

# Patterns to match common medical units
UNIT_PATTERN = re.compile(
    r'\b(\d+(?:\.\d+)?)\s*(g/dL|mg/dL|mmol/L|umol/L|uIU/mL|bpm|mmHg|pg|fL|%|mg|mcg|ml|mL|IU|g)\b',
    re.IGNORECASE
)

# Patterns to match dosage schedules
MED_SCHEDULE_PATTERN = re.compile(
    r'\b(?:1-0-1|1-1-1|0-0-1|1-0-0|0-1-0|once daily|twice daily|thrice daily|at bedtime|before food|after food|daily)\b',
    re.IGNORECASE
)

# Basic patient info patterns
PATIENT_PATTERN = re.compile(r'(?:patient\s*name\s*:?|patient\s*:?)\s*([a-zA-Z\s\.\-_]+)', re.IGNORECASE)
DOCTOR_PATTERN = re.compile(r'(?:doctor\s*:?|dr\.\s*)\s*([a-zA-Z\s\.\-_]+)', re.IGNORECASE)
DATE_PATTERN = re.compile(r'\b\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}\b')

def extract_structured_summary(text: str, doc_type: str) -> str:
    """
    Scans the extracted text and gathers structured metadata and key findings.
    """
    lines = text.split("\n")
    summary_lines = []

    patient_name = None
    doctor_name = None
    dates = []

    # 1. Extract metadata
    for line in lines:
        if not patient_name:
            match = PATIENT_PATTERN.search(line)
            if match:
                patient_name = match.group(1).strip()
        if not doctor_name:
            match = DOCTOR_PATTERN.search(line)
            if match:
                doctor_name = match.group(1).strip()
        
        found_dates = DATE_PATTERN.findall(line)
        if found_dates:
            dates.extend(found_dates)

    if patient_name:
        summary_lines.append(f"- **Patient**: {patient_name}")
    if doctor_name:
        formatted_dr = doctor_name if doctor_name.lower().startswith("dr") else f"Dr. {doctor_name}"
        summary_lines.append(f"- **Doctor**: {formatted_dr}")
    if dates:
        unique_dates = list(dict.fromkeys(dates))
        summary_lines.append(f"- **Date(s)**: {', '.join(unique_dates)}")

    summary_lines.append("\n**Key Findings & Details:**")
    detected_findings = []

    # 2. Extract medical findings
    for line in lines:
        s_line = line.strip()
        if not s_line:
            continue

        if doc_type == "lab_report":
            if UNIT_PATTERN.search(s_line):
                detected_findings.append(f"- {s_line}")
        elif doc_type == "prescription":
            # Match dosage schedules or common prescription indicators
            if MED_SCHEDULE_PATTERN.search(s_line) or any(u in s_line.lower() for u in ["mg", "mcg", "tab", "capsule", "tablet", "syrup"]):
                detected_findings.append(f"- {s_line}")
        else:
            # Fallback for mixed or undefined documents
            if UNIT_PATTERN.search(s_line) or MED_SCHEDULE_PATTERN.search(s_line):
                detected_findings.append(f"- {s_line}")

    if detected_findings:
        # Deduplicate findings keeping order
        seen_findings = set()
        unique_findings = []
        for f in detected_findings:
            if f not in seen_findings:
                seen_findings.add(f)
                unique_findings.append(f)
        summary_lines.extend(unique_findings)
    else:
        summary_lines.append("- No specific structured medical details matched the heuristics.")

    return "\n".join(summary_lines)

def process_document(text: str, doc_type: str, base_metadata: dict, chunker) -> List[Chunk]:
    """
    Cleans raw text, extracts a structured summary, and chunks the document.
    Returns a list of Chunks starting with the structured summary chunk.
    """
    # 1. Clean raw text (remove excessive whitespace lines)
    cleaned_lines = [line.strip() for line in text.split("\n") if line.strip()]
    cleaned_text = "\n".join(cleaned_lines)

    if not cleaned_text:
        return []

    # 2. Extract structured summary
    summary_text = extract_structured_summary(cleaned_text, doc_type)
    
    # 3. Create structured summary chunk
    summary_metadata = {
        **base_metadata,
        "chunk_type": "structured_summary",
        "is_summary": True,
        "chunk_index": 0
    }
    summary_chunk = Chunk(
        text=f"[STRUCTURED DOCUMENT SUMMARY]\n{summary_text}",
        metadata=summary_metadata
    )

    # 4. Chunk the cleaned raw text normally
    raw_chunks = chunker.chunk(cleaned_text, base_metadata)
    
    # Adjust indexes of standard chunks
    for i, c in enumerate(raw_chunks):
        c.metadata["chunk_index"] = i + 1
        c.metadata["is_summary"] = False

    return [summary_chunk] + raw_chunks
