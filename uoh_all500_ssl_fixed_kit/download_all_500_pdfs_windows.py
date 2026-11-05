
"""
Windows-native downloader for all 500 University of Hyderabad PDFs.

Features:
- Uses Windows curl.exe with Schannel where available.
- SSL certificate verification remains enabled.
- Skips existing valid PDFs.
- Retries failed downloads.
- Validates PDF files.
- Generates a CSV download log.
- Creates one ZIP containing all PDFs and the log.
"""

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import subprocess
import re
import shutil
import time
import zipfile

ROOT = Path(__file__).resolve().parent

CSV_PATH = ROOT / "uoh_all500_source_urls.csv"
PDF_DIR = ROOT / "uoh_all500_source_pdfs"
LOG_PATH = ROOT / "uoh_all500_download_log.csv"
ZIP_PATH = ROOT / "uoh_all500_source_pdfs_and_log.zip"

WORKERS = 6
TIMEOUT = 50


def safe_name(value):
    return re.sub(r"[^A-Za-z0-9_-]", "", str(value)) or "unknown"


def curl_path():
    path = shutil.which("curl.exe") or shutil.which("curl")

    if not path:
        raise RuntimeError("curl.exe not found.")

    return path


def is_valid_pdf(path):
    try:
        if not path.exists() or path.stat().st_size <= 1000:
            return False

        with path.open("rb") as file:
            return file.read(5) == b"%PDF-"

    except OSError:
        return False


def download(row, curl):
    dest = PDF_DIR / (safe_name(row["notice_id"]) + ".pdf")
    url = row["source_url"]

    # Reuse previously downloaded valid PDF.
    if is_valid_pdf(dest):
        return {
            **row,
            "local_file": str(dest),
            "bytes": dest.stat().st_size,
            "status": "Already downloaded - valid PDF"
        }

    last_error = ""

    for attempt in range(2):
        try:
            command = [
                curl,
                "--fail",
                "--location",
                "--silent",
                "--show-error",
                "--max-time", str(TIMEOUT),
                "--retry", "0",
                "--output", str(dest),
                "--write-out", "%{http_code}",
                "--user-agent",
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AcademicResearch/1.0",
                url
            ]

            process = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=TIMEOUT + 10
            )

            if process.returncode == 0 and is_valid_pdf(dest):
                return {
                    **row,
                    "local_file": str(dest),
                    "bytes": dest.stat().st_size,
                    "status": "Downloaded valid PDF"
                }

            last_error = (
                process.stderr
                or f"curl exit={process.returncode}, response={process.stdout}"
            ).strip()

            if dest.exists():
                dest.unlink()

        except Exception as error:
            last_error = str(error)

            if dest.exists():
                dest.unlink()

        time.sleep(1 + attempt)

    return {
        **row,
        "local_file": "",
        "bytes": 0,
        "status": "Failed/unresolved: " + last_error[:400]
    }


def main():

    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"Missing input file: {CSV_PATH.name}"
        )

    curl = curl_path()
    PDF_DIR.mkdir(exist_ok=True)

    with CSV_PATH.open(
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as file:
        rows = list(csv.DictReader(file))

    required_columns = {"notice_id", "source_url"}

    if not rows or not required_columns.issubset(rows[0].keys()):
        raise ValueError(
            "CSV must contain notice_id and source_url columns."
        )

    print(f"Using curl: {curl}")
    print(f"Total source URLs: {len(rows)}")
    print(f"Parallel workers: {WORKERS}")
    print("SSL certificate verification: ENABLED")
    print("-" * 60)

    results = []

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:

        futures = [
            pool.submit(download, row, curl)
            for row in rows
        ]

        for index, future in enumerate(as_completed(futures), 1):

            item = future.result()
            results.append(item)

            print(
                f"[{index}/{len(rows)}] "
                f"{item['notice_id']} - {item['status']}"
            )

    # Preserve original CSV order.
    order = {
        row["notice_id"]: index
        for index, row in enumerate(rows)
    }

    results.sort(
        key=lambda item: order.get(item["notice_id"], 999999)
    )

    # Save download log.
    with LOG_PATH.open(
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "notice_id",
                "title",
                "source_url",
                "local_file",
                "bytes",
                "status"
            ],
            extrasaction="ignore"
        )

        writer.writeheader()
        writer.writerows(results)

    # Create ZIP archive.
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()

    with zipfile.ZipFile(
        ZIP_PATH,
        "w",
        compression=zipfile.ZIP_DEFLATED
    ) as archive:

        archive.write(
            LOG_PATH,
            LOG_PATH.name
        )

        for item in results:

            if item["local_file"]:

                pdf_path = Path(item["local_file"])

                if is_valid_pdf(pdf_path):
                    archive.write(
                        pdf_path,
                        Path("uoh_all500_source_pdfs") / pdf_path.name
                    )

    # Final statistics.
    successful = sum(
        bool(item["local_file"]) and is_valid_pdf(Path(item["local_file"]))
        for item in results
    )

    failed = len(rows) - successful

    print("\n" + "=" * 60)
    print("DOWNLOAD AND ZIP PROCESS COMPLETE")
    print("=" * 60)

    print(f"URLs attempted: {len(rows)}")
    print(f"Valid PDFs available: {successful}")
    print(f"Failed/unresolved: {failed}")
    print(f"Download log: {LOG_PATH}")
    print(f"ZIP archive: {ZIP_PATH}")

    print("\nUpload this ZIP for the next dataset audit.")


if __name__ == "__main__":
    main()
