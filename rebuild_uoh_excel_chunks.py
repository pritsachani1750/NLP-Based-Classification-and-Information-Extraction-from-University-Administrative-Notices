#!/usr/bin/env python3
"""
Rebuild the University of Hyderabad dataset Excel workbook without truncating
long extracted PDF text.

Excel cells have a 32,767-character limit. This script:
  - Reads uoh_notice_candidates_repaired.csv if present, otherwise
    uoh_notice_candidates.csv.
  - Writes all metadata to a Candidates sheet.
  - Places a short text preview and full text character count in Candidates.
  - Preserves the COMPLETE extracted text in Text_Chunks, split into
    30,000-character pieces (each below Excel's cell limit).
  - Writes a Long_Text_Audit sheet for unusually long extracted text.

Install:
  python -m pip install pandas openpyxl

Run:
  python rebuild_uoh_excel_chunks.py
"""

from pathlib import Path
import pandas as pd

REPAIRED_CSV = Path("uoh_notice_candidates_repaired.csv")
ORIGINAL_CSV = Path("uoh_notice_candidates.csv")
OUTPUT_XLSX = Path("uoh_notice_candidates_repaired_fulltext.xlsx")

CHUNK_SIZE = 30000
PREVIEW_SIZE = 1000


def main():
    input_csv = REPAIRED_CSV if REPAIRED_CSV.exists() else ORIGINAL_CSV
    if not input_csv.exists():
        raise SystemExit(
            "Could not find uoh_notice_candidates_repaired.csv or "
            "uoh_notice_candidates.csv in this folder."
        )

    df = pd.read_csv(input_csv, keep_default_na=False)
    if "notice_text" not in df.columns:
        raise SystemExit("The CSV does not contain a notice_text column.")

    # Keep the complete source text in memory; do not modify the CSV.
    full_texts = df["notice_text"].astype(str).tolist()
    candidates = df.drop(columns=["notice_text"]).copy()
    candidates.insert(0, "source_row_number", range(1, len(candidates) + 1))
    candidates["notice_text_char_count"] = [len(t) for t in full_texts]
    candidates["notice_text_preview"] = [
        t[:PREVIEW_SIZE] for t in full_texts
    ]
    candidates["full_text_location"] = [
        "Text_Chunks sheet; match notice_id and chunk_index"
        for _ in full_texts
    ]

    chunk_rows = []
    audit_rows = []

    ids = (
        df["notice_id"].astype(str).tolist()
        if "notice_id" in df.columns
        else [f"ROW-{i:04d}" for i in range(1, len(df) + 1)]
    )
    titles = (
        df["title_from_archive"].astype(str).tolist()
        if "title_from_archive" in df.columns
        else [""] * len(df)
    )

    for row_num, (notice_id, title, full_text) in enumerate(
        zip(ids, titles, full_texts), start=1
    ):
        if len(full_text) > 32767:
            audit_rows.append({
                "notice_id": notice_id,
                "title": title,
                "text_char_count": len(full_text),
                "excel_cell_limit": 32767,
                "handling": "Full text preserved in Text_Chunks sheet"
            })

        if not full_text:
            chunk_rows.append({
                "notice_id": notice_id,
                "title": title,
                "chunk_index": 1,
                "chunk_count": 1,
                "text_chunk": ""
            })
            continue

        chunks = [
            full_text[i:i + CHUNK_SIZE]
            for i in range(0, len(full_text), CHUNK_SIZE)
        ]
        for chunk_index, chunk in enumerate(chunks, start=1):
            chunk_rows.append({
                "notice_id": notice_id,
                "title": title,
                "chunk_index": chunk_index,
                "chunk_count": len(chunks),
                "text_chunk": chunk
            })

    text_chunks = pd.DataFrame(chunk_rows)
    long_audit = pd.DataFrame(
        audit_rows,
        columns=[
            "notice_id", "title", "text_char_count",
            "excel_cell_limit", "handling"
        ]
    )

    with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
        candidates.to_excel(writer, index=False, sheet_name="Candidates")
        text_chunks.to_excel(writer, index=False, sheet_name="Text_Chunks")
        long_audit.to_excel(writer, index=False, sheet_name="Long_Text_Audit")

        for sheet_name in writer.sheets:
            ws = writer.sheets[sheet_name]
            ws.freeze_panes = "A2"
            ws.auto_filter.ref = ws.dimensions
            for cells in ws.columns:
                letter = cells[0].column_letter
                header = str(cells[0].value or "")
                if header == "text_chunk":
                    width = 60
                elif header in ("notice_text_preview",):
                    width = 70
                else:
                    width = min(
                        max(max(len(str(c.value or "")) for c in cells) + 2, 14),
                        55
                    )
                ws.column_dimensions[letter].width = width

    print(f"Input CSV: {input_csv.resolve()}")
    print(f"Rows: {len(df)}")
    print(f"Text chunks: {len(text_chunks)}")
    print(f"Records with text over Excel cell limit: {len(long_audit)}")
    print(f"Excel workbook created: {OUTPUT_XLSX.resolve()}")
    print("The original CSV retains the full extracted text in one field.")
    print("The Excel Text_Chunks sheet preserves long text across multiple cells.")


if __name__ == "__main__":
    main()
