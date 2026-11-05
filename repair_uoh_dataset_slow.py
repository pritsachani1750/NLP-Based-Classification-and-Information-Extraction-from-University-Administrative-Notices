#!/usr/bin/env python3
"""
University of Hyderabad candidate dataset – slow/reliable download repair.

Run this script in the folder containing:
  uoh_notice_candidates.csv

Install:
  python -m pip install truststore requests pandas pymupdf openpyxl

Run:
  python repair_uoh_dataset_slow.py

Design:
  * SSL certificate verification remains enabled.
  * Downloads are strictly sequential (no parallel requests).
  * Each attempt uses a fresh HTTP connection (Connection: close).
  * Exponential backoff: 15, 30, 60, 120, 180 seconds.
  * Saves CSV and Excel after each processed row.
  * Does not fabricate text, assign categories, or mark records verified.
"""

from pathlib import Path
import re
import sys
import time

import truststore
truststore.inject_into_ssl()

import pandas as pd
import requests
import pymupdf

INPUT_CSV = Path("uoh_notice_candidates.csv")
OUTPUT_CSV = Path("uoh_notice_candidates_repaired.csv")
OUTPUT_XLSX = Path("uoh_notice_candidates_repaired.xlsx")

MAX_ATTEMPTS = 5
BACKOFF_SECONDS = [15, 30, 60, 120, 180]
DELAY_BETWEEN_FILES = 8
TIMEOUT = (30, 120)


def clean_text(value):
    return re.sub(r"\s+", " ", value or "").strip()


def extract_pdf_text(path):
    with pymupdf.open(path) as doc:
        page_count = len(doc)
        text = clean_text(" ".join(page.get_text("text") for page in doc))
    return text, page_count


def is_retryable_failure(status):
    value = str(status or "").lower()
    return any(term in value for term in (
        "request_error", "ssl_error", "download_error",
        "connectionerror", "connectionreset", "timeout",
        "processing_error", "retry_failed"
    ))


def save_files(df):
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Candidates")
        ws = writer.sheets["Candidates"]
        ws.freeze_panes = "A2"
        ws.auto_filter.ref = ws.dimensions
        for cells in ws.columns:
            letter = cells[0].column_letter
            width = min(max(max(len(str(c.value or "")) for c in cells) + 2, 14), 55)
            ws.column_dimensions[letter].width = width


def get_pdf(url):
    """Try one fresh HTTPS connection. Certificate verification is ON."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 Chrome/124.0 Safari/537.36"
        ),
        "Accept": "application/pdf,*/*",
        "Connection": "close",
    }
    response = requests.get(url, headers=headers, timeout=TIMEOUT, stream=True)
    response.raise_for_status()

    chunks = []
    for chunk in response.iter_content(chunk_size=1024 * 128):
        if chunk:
            chunks.append(chunk)
    content = b"".join(chunks)
    response.close()

    if not content.startswith(b"%PDF"):
        raise ValueError(
            "Response is not a PDF; content-type was "
            + str(response.headers.get("Content-Type", "unknown"))
        )
    return content


def main():
    if not INPUT_CSV.exists():
        sys.exit(
            f"Input file not found: {INPUT_CSV.resolve()}\n"
            "Place this script beside uoh_notice_candidates.csv."
        )

    df = pd.read_csv(INPUT_CSV, keep_default_na=False)
    required = {"source_url", "local_pdf", "download_status", "notice_text"}
    missing = required - set(df.columns)
    if missing:
        sys.exit(f"CSV is missing required columns: {sorted(missing)}")

    failed_indices = [
        idx for idx, row in df.iterrows()
        if is_retryable_failure(row["download_status"])
    ]

    print(f"Rows in input: {len(df)}")
    print(f"Rows selected for retry: {len(failed_indices)}")
    print("Sequential mode; long waits are intentional to avoid repeated resets.")
    print("Existing successfully extracted text will be preserved.")

    for position, idx in enumerate(failed_indices, start=1):
        row = df.loc[idx]
        url = str(row["source_url"]).strip()
        if not url:
            df.at[idx, "download_status"] = "retry_skipped_missing_source_url"
            save_files(df)
            continue

        raw_path = str(row["local_pdf"]).strip()
        pdf_path = Path(raw_path) if raw_path else (
            Path("uoh_collection_500") / "pdfs" / f"retry_{idx}.pdf"
        )
        pdf_path.parent.mkdir(parents=True, exist_ok=True)

        # If an existing local file is a valid PDF, extract it before networking.
        if pdf_path.exists() and pdf_path.stat().st_size > 0:
            try:
                text, pages = extract_pdf_text(pdf_path)
                if text:
                    df.at[idx, "notice_text"] = text
                    df.at[idx, "page_count"] = pages
                    df.at[idx, "local_pdf"] = str(pdf_path.resolve())
                    df.at[idx, "download_status"] = "existing_pdf_text_extracted"
                    save_files(df)
                    print(f"[{position}/{len(failed_indices)}] Reused local PDF: {pdf_path.name}")
                    continue
            except Exception:
                pass

        succeeded = False
        last_error = ""

        print(f"\n[{position}/{len(failed_indices)}] {url}")

        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                content = get_pdf(url)
                pdf_path.write_bytes(content)
                text, pages = extract_pdf_text(pdf_path)

                df.at[idx, "local_pdf"] = str(pdf_path.resolve())
                df.at[idx, "page_count"] = pages
                df.at[idx, "notice_text"] = text
                df.at[idx, "download_status"] = (
                    "retried_download_text_extracted"
                    if text else
                    "retried_download_no_selectable_text_OCR_may_be_needed"
                )
                succeeded = True
                print(f"  Downloaded: {len(content):,} bytes; pages={pages}; text chars={len(text)}")
                break

            except KeyboardInterrupt:
                print("\nInterrupted by user. Saving completed work before exit.")
                save_files(df)
                raise
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                print(f"  Attempt {attempt}/{MAX_ATTEMPTS} failed: {last_error}")
                if attempt < MAX_ATTEMPTS:
                    wait = BACKOFF_SECONDS[attempt - 1]
                    print(f"  Waiting {wait} seconds before retry...")
                    time.sleep(wait)

        if not succeeded:
            df.at[idx, "download_status"] = f"retry_failed: {last_error}"
            print("  Could not retrieve this PDF; it remains in the dataset for later retry.")

        save_files(df)
        time.sleep(DELAY_BETWEEN_FILES)

    status = df["download_status"].astype(str)
    text_present = df["notice_text"].astype(str).str.strip().ne("")
    still_failed = status.str.contains(
        r"request_error|ssl_error|download_error|retry_failed|connectionreset",
        case=False, regex=True
    )

    print("\n========== REPAIR SUMMARY ==========")
    print(f"Total candidate records: {len(df)}")
    print(f"Records with extracted text: {int(text_present.sum())}")
    print(f"Records still without text: {int((~text_present).sum())}")
    print(f"Records still marked failed: {int(still_failed.sum())}")
    print(f"CSV: {OUTPUT_CSV.resolve()}")
    print(f"Excel: {OUTPUT_XLSX.resolve()}")
    print("\nNo annotations were invented; all records still require human review.")


if __name__ == "__main__":
    main()
