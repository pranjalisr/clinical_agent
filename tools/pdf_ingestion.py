"""
PDF Ingestion Tool
Extracts text from clinical PDF documents using OCR.
Handles both text-based and scanned PDFs.
"""

import os
import time
from pathlib import Path
from typing import Optional
import subprocess

try:
    from pdf2image import convert_from_path
    import pytesseract
    from PIL import Image
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False


def extract_text_from_pdf(pdf_path: str, max_pages: Optional[int] = None) -> dict:
    """
    Extract text from a PDF file.
    Returns a dict with:
      - success: bool
      - text: str (full extracted text)
      - pages: list of per-page text
      - method: str (how it was extracted)
      - error: str (if failed)
      - page_count: int
    """
    result = {
        "success": False,
        "text": "",
        "pages": [],
        "method": "unknown",
        "error": None,
        "page_count": 0,
        "pdf_path": pdf_path,
    }

    if not os.path.exists(pdf_path):
        result["error"] = f"File not found: {pdf_path}"
        return result

    # Step 1: Try native text extraction via pdftotext
    try:
        proc = subprocess.run(
            ["pdftotext", pdf_path, "-"],
            capture_output=True, text=True, timeout=30
        )
        native_text = proc.stdout.strip()
        # pdftotext returns form-feeds for page breaks — check if we got real text
        pages_raw = native_text.split("\x0c")
        meaningful_pages = [p for p in pages_raw if len(p.strip()) > 30]
        if meaningful_pages:
            result["success"] = True
            result["text"] = "\n\n--- PAGE BREAK ---\n\n".join(meaningful_pages)
            result["pages"] = meaningful_pages
            result["method"] = "native_text"
            result["page_count"] = len(meaningful_pages)
            return result
    except Exception as e:
        pass  # Fall through to OCR

    # Step 2: OCR via pdf2image + pytesseract
    if not PDF2IMAGE_AVAILABLE:
        result["error"] = "pdf2image/pytesseract not installed. Run: pip install pdf2image pytesseract Pillow"
        return result

    try:
        # Get page count first
        info_proc = subprocess.run(
            ["pdfinfo", pdf_path], capture_output=True, text=True, timeout=10
        )
        total_pages = 1
        for line in info_proc.stdout.splitlines():
            if line.startswith("Pages:"):
                total_pages = int(line.split(":")[1].strip())
                break

        result["page_count"] = total_pages
        pages_to_process = min(total_pages, max_pages) if max_pages else total_pages

        all_pages_text = []
        # Process in small batches to avoid memory issues
        batch_size = 5
        for batch_start in range(1, pages_to_process + 1, batch_size):
            batch_end = min(batch_start + batch_size - 1, pages_to_process)
            images = convert_from_path(
                pdf_path,
                dpi=150,
                first_page=batch_start,
                last_page=batch_end,
                thread_count=1,
            )
            for img in images:
                page_text = pytesseract.image_to_string(img, config="--psm 6")
                all_pages_text.append(page_text.strip())

        # Filter out near-empty pages
        meaningful = [p for p in all_pages_text if len(p.strip()) > 20]

        result["success"] = True
        result["pages"] = all_pages_text
        result["text"] = "\n\n--- PAGE BREAK ---\n\n".join(all_pages_text)
        result["method"] = "ocr"
        result["page_count"] = total_pages

    except Exception as e:
        result["error"] = f"OCR failed: {str(e)}"

    return result

def _read_text_file(path: str) -> dict:
    """Read a plain text file directly — no OCR needed."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        return {
            "success": True,
            "text": text,
            "pages": [text],
            "method": "plaintext",
            "error": None,
            "page_count": 1,
        }
    except Exception as e:
        return {"success": False, "text": "", "pages": [], "method": "plaintext", "error": str(e), "page_count": 0}

def ingest_patient_folder(folder_path: str) -> dict:
    """
    Ingest all PDFs in a patient folder.
    Returns dict mapping filename -> extraction result.
    """
    folder = Path(folder_path)
    if not folder.exists():
        return {"error": f"Folder not found: {folder_path}"}

    pdf_files = list(folder.glob("*.pdf")) + list(folder.glob("*.PDF"))
    txt_files = list(folder.glob("*.txt"))
    all_files = pdf_files + txt_files
    if not all_files:
        return {"error": f"No PDF or text files found in {folder_path}"}

    results = {}
    for doc_file in sorted(all_files):
        if doc_file.suffix.lower() == ".txt":
            results[doc_file.name] = _read_text_file(str(doc_file))
        else:
            results[doc_file.name] = extract_text_from_pdf(str(doc_file))

    return results
