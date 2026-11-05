#!/usr/bin/env python3
"""
University of Hyderabad – Official Notice PDF Collector
=======================================================

Purpose:
  1. Open the official University of Hyderabad "All Notices" archive.
  2. Click the archive's "Show more" control to reveal more notices.
  3. Collect up to 500 unique original PDF URLs.
  4. Download PDFs using Windows' trusted certificate store via truststore.
  5. Extract selectable PDF text using PyMuPDF.
  6. Save a candidate CSV and Excel workbook for manual verification.

IMPORTANT:
  - This script does NOT fabricate records.
  - It does NOT automatically mark notices as verified.
  - A downloaded PDF is a candidate record until a human checks its source,
    text, title, date, deadline, and category.
  - SSL certificate verification remains ENABLED. Do not set verify=False.

Install dependencies (PowerShell):
  pip install playwright requests pandas pymupdf openpyxl truststore
  python -m playwright install chromium

Run:
  python collect_uoh_notices_500_final.py

If Google Chrome is installed but Chromium download fails, set USE_CHROME = True
below and the script will try to use the installed Google Chrome browser.
"""

from __future__ import annotations

import hashlib
import re
import sys
import time
from pathlib import Path
from urllib.parse import urlparse, unquote

# Use the Windows/system certificate store for Python HTTPS verification.
# This addresses common CERTIFICATE_VERIFY_FAILED errors on Windows networks.
try:
    import truststore
    truststore.inject_into_ssl()
except ImportError:
    print(
        "ERROR: truststore is not installed.\n"
        "Run: pip install truststore"
    )
    raise

import pandas as pd
import requests
import pymupdf  # Current PyMuPDF API
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# ------------------------- CONFIGURATION -------------------------

ARCHIVE_URL = "https://www.uohyd.ac.in/or/all-notices/"
TARGET_RECORDS = 500

OUTPUT_DIR = Path("uoh_collection_500")
PDF_DIR = OUTPUT_DIR / "pdfs"

# False = use Playwright-managed Chromium.
# True = use installed Google Chrome (requires Chrome installed).
USE_CHROME = False

# Maximum number of "Show more" attempts.
MAX_LOAD_MORE_CLICKS = 100

# Small pause between PDF downloads, in seconds.
DOWNLOAD_DELAY = 0.25

# ----------------------------------------------------------------


def clean_text(value: str | None) -> str:
    """Normalize whitespace without changing the original language."""
    return re.sub(r"\s+", " ", value or "").strip()


def make_session() -> requests.Session:
    """Create a requests session with retries; SSL verification stays enabled."""
    session = requests.Session()
    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0 Safari/537.36"
        )
    })

    retry = Retry(
        total=3,
        connect=3,
        read=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def is_pdf_url(url: str) -> bool:
    """Return True for links whose URL path ends in .pdf."""
    return bool(re.search(r"\.pdf(?:$|[?#])", url, flags=re.IGNORECASE))


def collect_pdf_links() -> list[dict[str, str]]:
    """
    Use a real browser to expand the dynamic archive and collect PDF links.
    Returns unique URLs in the order encountered.
    """
    print("\n[1/3] Opening the official University of Hyderabad archive...")
    browser = None

    with sync_playwright() as playwright:
        launch_options = {"headless": True}

        if USE_CHROME:
            launch_options["channel"] = "chrome"

        try:
            browser = playwright.chromium.launch(**launch_options)
        except Exception as exc:
            print("\nCould not launch the selected browser.")
            if USE_CHROME:
                print("Check that Google Chrome is installed.")
            else:
                print(
                    "Install Playwright Chromium with:\n"
                    "python -m playwright install chromium"
                )
            raise exc

        page = browser.new_page()
        page.set_default_timeout(15000)

        try:
            page.goto(
                ARCHIVE_URL,
                wait_until="domcontentloaded",
                timeout=90000
            )
            page.wait_for_timeout(2500)

            previous_count = -1
            unchanged_rounds = 0

            for click_number in range(1, MAX_LOAD_MORE_CLICKS + 1):
                current_links = page.locator("a[href]").evaluate_all(
                    """elements => elements.map(a => ({
                        href: a.href,
                        title: (a.innerText || a.textContent || '').trim(),
                        context: (a.parentElement?.innerText || '').trim()
                    }))"""
                )

                current_pdf_count = len({
                    item["href"]
                    for item in current_links
                    if is_pdf_url(item.get("href", ""))
                })

                print(
                    f"  Archive expansion {click_number}: "
                    f"{current_pdf_count} unique PDF links visible"
                )

                if current_pdf_count >= TARGET_RECORDS:
                    print("  Target number of PDF links reached.")
                    break

                if current_pdf_count == previous_count:
                    unchanged_rounds += 1
                else:
                    unchanged_rounds = 0

                if unchanged_rounds >= 3:
                    print("  No additional PDF links appeared; stopping expansion.")
                    break

                previous_count = current_pdf_count

                # Find a visible control labelled "Show more".
                show_more = page.get_by_text(
                    re.compile(r"^\s*show\s+more\s*$", re.IGNORECASE)
                )

                try:
                    visible_count = show_more.count()
                    if visible_count == 0:
                        # A looser match accommodates buttons with extra text.
                        show_more = page.get_by_text(
                            re.compile(r"show\s+more", re.IGNORECASE)
                        )
                        visible_count = show_more.count()

                    if visible_count == 0:
                        print("  No 'Show more' control found; stopping.")
                        break

                    show_more.first.click(timeout=10000)
                    page.wait_for_timeout(1400)

                except PlaywrightTimeoutError:
                    print("  'Show more' did not respond; stopping expansion.")
                    break

            # Collect final page links after expansion.
            final_links = page.locator("a[href]").evaluate_all(
                """elements => elements.map(a => ({
                    href: a.href,
                    title: (a.innerText || a.textContent || '').trim(),
                    context: (a.parentElement?.innerText || '').trim()
                }))"""
            )

        finally:
            browser.close()

    # Deduplicate by exact URL, preserving first occurrence.
    unique: dict[str, dict[str, str]] = {}
    for item in final_links:
        href = item.get("href", "").strip()
        if href and is_pdf_url(href) and href not in unique:
            unique[href] = item

    results = list(unique.values())[:TARGET_RECORDS]
    print(f"\n  Total unique PDF URLs selected: {len(results)}")
    return results


def safe_filename(url: str, index: int) -> str:
    """Create a safe unique filename based on the PDF URL."""
    path_name = unquote(Path(urlparse(url).path).name)
    if not path_name.lower().endswith(".pdf"):
        path_name = f"notice_{index}.pdf"

    path_name = re.sub(r'[<>:"/\\|?*]', "_", path_name)
    # Add a short URL hash to prevent collisions between same-named PDFs.
    url_hash = hashlib.sha1(url.encode("utf-8")).hexdigest()[:8]
    stem = Path(path_name).stem[:100]
    return f"{stem}_{url_hash}.pdf"


def extract_pdf_text(pdf_path: Path) -> tuple[str, int]:
    """Extract selectable text from all pages of a PDF."""
    with pymupdf.open(pdf_path) as document:
        page_count = len(document)
        text_parts = [page.get_text("text") for page in document]
    return clean_text("\n".join(text_parts)), page_count


def download_one(
    session: requests.Session,
    url: str,
    destination: Path
) -> tuple[str, str, int]:
    """
    Download one PDF and extract text.
    Returns (download_status, extracted_text, page_count).
    """
    try:
        response = session.get(url, timeout=(20, 90))
        response.raise_for_status()

        # Check PDF signature rather than trusting only the HTTP content-type.
        if not response.content.startswith(b"%PDF"):
            content_type = response.headers.get("Content-Type", "")
            raise ValueError(
                f"URL did not return a PDF (Content-Type: {content_type})"
            )

        destination.write_bytes(response.content)
        text, page_count = extract_pdf_text(destination)

        if text:
            status = "downloaded_text_extracted"
        else:
            status = "downloaded_no_selectable_text_OCR_may_be_needed"

        return status, text, page_count

    except requests.exceptions.SSLError as exc:
        return f"ssl_error: {exc}", "", 0
    except requests.exceptions.RequestException as exc:
        return f"request_error: {exc}", "", 0
    except Exception as exc:
        return f"processing_error: {type(exc).__name__}: {exc}", "", 0


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PDF_DIR.mkdir(parents=True, exist_ok=True)

    links = collect_pdf_links()

    if not links:
        print("\nNo PDF links were found. The website layout may have changed.")
        sys.exit(1)

    print("\n[2/3] Downloading PDFs and extracting text...")
    session = make_session()
    rows: list[dict[str, object]] = []

    for index, item in enumerate(links, start=1):
        url = item.get("href", "").strip()
        title = clean_text(item.get("title")) or clean_text(
            item.get("context")
        )[:300]

        filename = safe_filename(url, index)
        local_pdf = PDF_DIR / filename

        # Resume-friendly behaviour: reuse an existing PDF and extract again.
        if local_pdf.exists() and local_pdf.stat().st_size > 0:
            try:
                text, page_count = extract_pdf_text(local_pdf)
                status = (
                    "existing_pdf_text_extracted"
                    if text else
                    "existing_pdf_no_selectable_text_OCR_may_be_needed"
                )
            except Exception:
                status, text, page_count = download_one(
                    session, url, local_pdf
                )
        else:
            status, text, page_count = download_one(
                session, url, local_pdf
            )

        row = {
            "notice_id": hashlib.sha1(url.encode("utf-8")).hexdigest()[:12],
            "title_from_archive": title,
            "notice_text": text,
            "source_url": url,
            "archive_url": ARCHIVE_URL,
            "local_pdf": str(local_pdf.resolve()) if local_pdf.exists() else "",
            "page_count": page_count,
            "category": "",
            "issuing_office": "",
            "archive_publication_date": "",
            "document_issue_date": "",
            "deadline": "",
            "reference_number": "",
            "email_addresses": "",
            "verification_status": "needs_human_review",
            "download_status": status,
            "annotation_notes": "",
        }
        rows.append(row)

        print(
            f"[{index}/{len(links)}] {status} | "
            f"{title[:100] or url}"
        )

        time.sleep(DOWNLOAD_DELAY)

        # Save progress every 25 records so an interrupted run is recoverable.
        if index % 25 == 0 or index == len(links):
            partial = pd.DataFrame(rows)
            partial.to_csv(
                OUTPUT_DIR / "uoh_notice_candidates.csv",
                index=False,
                encoding="utf-8-sig"
            )
            partial.to_excel(
                OUTPUT_DIR / "uoh_notice_candidates.xlsx",
                index=False,
                sheet_name="Candidates"
            )
            print(f"  Progress saved: {index}/{len(links)} records")

    print("\n[3/3] Preparing final candidate files...")
    dataframe = pd.DataFrame(rows)

    csv_path = OUTPUT_DIR / "uoh_notice_candidates.csv"
    excel_path = OUTPUT_DIR / "uoh_notice_candidates.xlsx"

    dataframe.to_csv(csv_path, index=False, encoding="utf-8-sig")

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        dataframe.to_excel(writer, index=False, sheet_name="Candidates")
        worksheet = writer.sheets["Candidates"]
        worksheet.freeze_panes = "A2"
        worksheet.auto_filter.ref = worksheet.dimensions

        for column_cells in worksheet.columns:
            column_letter = column_cells[0].column_letter
            max_length = max(
                len(str(cell.value or "")) for cell in column_cells
            )
            worksheet.column_dimensions[column_letter].width = min(
                max(max_length + 2, 14), 55
            )

    successful = dataframe["download_status"].astype(str).str.contains(
        "downloaded|existing_pdf", regex=True
    ).sum()
    extracted = dataframe["download_status"].astype(str).str.contains(
        "text_extracted", regex=False
    ).sum()
    errors = len(dataframe) - successful

    print("\n" + "=" * 60)
    print("COLLECTION SUMMARY")
    print("=" * 60)
    print(f"Unique PDF URLs selected : {len(dataframe)}")
    print(f"PDFs downloaded/reused   : {successful}")
    print(f"Records with extracted text: {extracted}")
    print(f"Download/processing errors: {errors}")
    print(f"CSV file                 : {csv_path.resolve()}")
    print(f"Excel file               : {excel_path.resolve()}")
    print(f"PDF folder               : {PDF_DIR.resolve()}")
    print("=" * 60)
    print(
        "\nIMPORTANT: All rows remain 'needs_human_review'. "
        "Inspect source PDFs and verify titles, dates, deadlines, "
        "issuing offices, and categories before academic submission."
    )
    print(
        "Scanned PDFs may require OCR. Corrigenda and duplicate notices "
        "should be reviewed and handled transparently."
    )


if __name__ == "__main__":
    main()
