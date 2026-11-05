"""
OCR recovery for the 45 University of Hyderabad PDFs identified in the PDF audit.

Run from the folder containing:
  uoh_all500_source_pdfs/
  uoh_all500_download_log.csv

Outputs:
  uoh_ocr_45_output/text/<notice_id>.txt
  uoh_ocr_45_results.csv
  uoh_ocr_45_combined.csv

Original PDFs are never modified. The script is resumable: completed OCR text
files are reused on the next run.
"""

from pathlib import Path
import csv
import shutil
import sys
import time

import fitz  # PyMuPDF
import pytesseract
from PIL import Image
from io import BytesIO

ROOT = Path(__file__).resolve().parent
PDF_DIR = ROOT / "uoh_all500_source_pdfs"
LOG_CSV = ROOT / "uoh_all500_download_log.csv"
OUT_DIR = ROOT / "uoh_ocr_45_output"
TEXT_DIR = OUT_DIR / "text"
RESULTS_CSV = ROOT / "uoh_ocr_45_results.csv"
COMBINED_CSV = ROOT / "uoh_ocr_45_combined.csv"

# These IDs are taken from the "Needs OCR Review" sheet of the audit report.
OCR_NOTICE_IDS = [
    "30ee6e368584", "274f478c2edd", "99bbf3dd92f0", "cb3bf92aad64",
    "b54c92dbf8e9", "a58a3b547e99", "7790ece55f33", "5796614bd856",
    "3a9b78864f99", "68e7244c2983", "c9e4c77938fa", "1ab9355e3c10",
    "21043707f76f", "fcb6c94deefd", "8af41a4b92ac", "d303791c1884",
    "383622f11b64", "8ef5b0f81ead", "f2f07494e449", "c0efc9fca504",
    "4ef1188ef3bc", "c708fb38d5ce", "4d9454178ae6", "b791b348ef6b",
    "41c94d0ad098", "ed24570e622d", "38f7af56d12c", "47cfb88a5376",
    "53568792d683", "7e2e171bb868", "53f98f814e72", "2a64dd264e6b",
    "866b5ebe0f38", "008346326cb6", "f800486b3b79", "815abece424c",
    "1b447e20ee19", "736bd80477bb", "019aaddb05a1", "d1e2c62d0cde",
    "d460f58d3b85", "0cebfb06c292", "5ba107e0aa2e", "536eb17bcc4e",
    "bf5f8513e666",
]

DPI = 220
LANG = "eng"
TESSERACT_CONFIG = "--oem 1 --psm 3"


def find_tesseract():
    found = shutil.which("tesseract")
    if found:
        pytesseract.pytesseract.tesseract_cmd = found
        return found

    common = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
    if common.exists():
        pytesseract.pytesseract.tesseract_cmd = str(common)
        return str(common)

    raise FileNotFoundError(
        "Tesseract was not found. Install it or add its folder to Windows PATH."
    )


def read_download_log():
    info = {}
    if LOG_CSV.exists():
        with LOG_CSV.open("r", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                info[row.get("notice_id", "")] = row
    return info


def ocr_pdf(pdf_path):
    page_texts = []
    total_pages = 0
    ocr_pages = 0

    with fitz.open(pdf_path) as doc:
        total_pages = len(doc)
        for page_number, page in enumerate(doc, start=1):
            # These 45 files were flagged for no extractable text by the audit.
            # Still preserve any native text found on an individual page.
            native_text = page.get_text("text").strip()

            if len(native_text) >= 30:
                extracted = native_text
                method = "native text"
            else:
                pix = page.get_pixmap(
                    matrix=fitz.Matrix(DPI / 72, DPI / 72),
                    colorspace=fitz.csRGB,
                    alpha=False
                )
                image = Image.open(BytesIO(pix.tobytes("png")))
                extracted = pytesseract.image_to_string(
                    image,
                    lang=LANG,
                    config=TESSERACT_CONFIG
                ).strip()
                method = "OCR"
                ocr_pages += 1

            page_texts.append(
                f"===== PAGE {page_number} ({method}) =====\n{extracted}"
            )

    return "\n\n".join(page_texts), total_pages, ocr_pages


def main():
    tesseract = find_tesseract()
    if not PDF_DIR.exists():
        raise FileNotFoundError(
            f"PDF folder not found: {PDF_DIR}\n"
            "Extract the downloaded ZIP so this script sits beside the PDF folder."
        )

    TEXT_DIR.mkdir(parents=True, exist_ok=True)
    log_info = read_download_log()
    results = []

    print(f"Tesseract: {tesseract}")
    print(f"PDF folder: {PDF_DIR}")
    print(f"Documents queued: {len(OCR_NOTICE_IDS)}")
    print(f"Rendering resolution: {DPI} DPI")
    print("Original PDFs will not be modified.\n")

    for index, notice_id in enumerate(OCR_NOTICE_IDS, start=1):
        pdf_path = PDF_DIR / f"{notice_id}.pdf"
        text_path = TEXT_DIR / f"{notice_id}.txt"
        source_row = log_info.get(notice_id, {})

        if not pdf_path.exists():
            result = {
                "notice_id": notice_id,
                "source_url": source_row.get("source_url", ""),
                "pdf_file": str(pdf_path),
                "page_count": "",
                "ocr_pages": "",
                "extracted_characters": 0,
                "status": "PDF file missing",
                "text_file": ""
            }
            results.append(result)
            print(f"[{index}/{len(OCR_NOTICE_IDS)}] {notice_id}: PDF MISSING")
            continue

        try:
            if text_path.exists() and text_path.stat().st_size > 0:
                text = text_path.read_text(encoding="utf-8", errors="replace")
                with fitz.open(pdf_path) as doc:
                    page_count = len(doc)
                ocr_pages = "reused"
                status = "Reused existing OCR output"
            else:
                text, page_count, ocr_pages = ocr_pdf(pdf_path)
                text_path.write_text(text, encoding="utf-8")
                status = "OCR completed" if len(text.strip()) >= 30 else "OCR produced little/no text"

            result = {
                "notice_id": notice_id,
                "source_url": source_row.get("source_url", ""),
                "pdf_file": str(pdf_path),
                "page_count": page_count,
                "ocr_pages": ocr_pages,
                "extracted_characters": len(text.strip()),
                "status": status,
                "text_file": str(text_path)
            }
            results.append(result)
            print(
                f"[{index}/{len(OCR_NOTICE_IDS)}] {notice_id}: "
                f"{status} | pages={page_count} | chars={len(text.strip())}"
            )

        except Exception as exc:
            results.append({
                "notice_id": notice_id,
                "source_url": source_row.get("source_url", ""),
                "pdf_file": str(pdf_path),
                "page_count": "",
                "ocr_pages": "",
                "extracted_characters": 0,
                "status": f"ERROR: {type(exc).__name__}: {exc}",
                "text_file": ""
            })
            print(f"[{index}/{len(OCR_NOTICE_IDS)}] {notice_id}: ERROR - {exc}")

        # Save a checkpoint after every document so interrupted runs can resume.
        write_results(results)

    write_results(results)
    write_combined(results)

    successful = sum(
        r["status"] in ("OCR completed", "Reused existing OCR output")
        for r in results
    )
    print("\nOCR RUN FINISHED")
    print(f"Documents in queue: {len(OCR_NOTICE_IDS)}")
    print(f"Completed/reused: {successful}")
    print(f"Needs attention: {len(results) - successful}")
    print(f"Detailed report: {RESULTS_CSV}")
    print(f"Combined text CSV: {COMBINED_CSV}")
    print(f"Page-wise text files: {TEXT_DIR}")


def write_results(results):
    fields = [
        "notice_id", "source_url", "pdf_file", "page_count",
        "ocr_pages", "extracted_characters", "status", "text_file"
    ]
    with RESULTS_CSV.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)


def write_combined(results):
    fields = ["notice_id", "source_url", "ocr_text", "status"]
    with COMBINED_CSV.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in results:
            text_file = row.get("text_file", "")
            text = ""
            if text_file and Path(text_file).exists():
                text = Path(text_file).read_text(
                    encoding="utf-8", errors="replace"
                )
            writer.writerow({
                "notice_id": row.get("notice_id", ""),
                "source_url": row.get("source_url", ""),
                "ocr_text": text,
                "status": row.get("status", "")
            })


if __name__ == "__main__":
    main()
