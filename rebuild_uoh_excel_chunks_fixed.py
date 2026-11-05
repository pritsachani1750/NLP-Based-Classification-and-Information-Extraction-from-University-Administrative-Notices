import re
from pathlib import Path

import pandas as pd

# Input preference: repaired file first, otherwise the original collector output.
INPUT_CANDIDATES = [
    Path("uoh_notice_candidates_repaired.csv"),
    Path("uoh_notice_candidates.csv"),
]
OUTPUT_XLSX = Path("uoh_notice_candidates_repaired_fulltext.xlsx")

CHUNK_SIZE = 30000  # Safely below Excel's 32,767-character cell limit.
PREVIEW_SIZE = 1000

# Excel/XML 1.0 does not permit most control characters.
ILLEGAL_XML_RE = re.compile(
    r"[\x00-\x08\x0B\x0C\x0E-\x1F\uD800-\uDFFF\uFFFE\uFFFF]"
)

def clean_for_excel(value):
    """Remove characters that Excel/openpyxl cannot store."""
    if value is None:
        return ""
    if not isinstance(value, str):
        return value
    return ILLEGAL_XML_RE.sub("", value)

def main():
    input_file = next((p for p in INPUT_CANDIDATES if p.exists()), None)
    if input_file is None:
        raise FileNotFoundError(
            "Could not find uoh_notice_candidates_repaired.csv or "
            "uoh_notice_candidates.csv in this folder."
        )

    print(f"Reading: {input_file}")
    df = pd.read_csv(input_file, dtype=str, keep_default_na=False)

    # Identify the extracted-text column using common collector column names.
    possible_text_columns = [
        "full_text", "extracted_text", "text", "notice_text",
        "pdf_text", "content"
    ]
    text_col = next(
        (c for c in possible_text_columns if c in df.columns),
        None
    )
    if text_col is None:
        raise ValueError(
            "Could not identify the extracted text column. "
            f"Available columns: {list(df.columns)}"
        )

    # Clean every text field before sending anything to Excel.
    for col in df.columns:
        df[col] = df[col].map(clean_for_excel)

    candidate_rows = []
    chunk_rows = []
    audit_rows = []

    for row_index, row in df.iterrows():
        record = row.to_dict()
        full_text = record.get(text_col, "") or ""
        record_id = (
            record.get("notice_id")
            or record.get("id")
            or record.get("pdf_url")
            or record.get("source_url")
            or str(row_index + 1)
        )

        # Keep the main sheet compact and Excel-safe.
        candidate_record = dict(record)
        candidate_record[text_col] = full_text[:PREVIEW_SIZE]
        candidate_record["full_text_char_count"] = len(full_text)
        candidate_record["full_text_location"] = (
            f"Text_Chunks; record_id={record_id}"
        )
        candidate_rows.append(candidate_record)

        if len(full_text) > 32767:
            audit_rows.append({
                "record_id": record_id,
                "text_column": text_col,
                "full_text_char_count": len(full_text),
                "excel_cell_limit": 32767,
                "note": "Full text split across Text_Chunks rows."
            })

        if not full_text:
            chunk_rows.append({
                "record_id": record_id,
                "chunk_number": 1,
                "text_chunk": ""
            })
        else:
            for start in range(0, len(full_text), CHUNK_SIZE):
                chunk_rows.append({
                    "record_id": record_id,
                    "chunk_number": (start // CHUNK_SIZE) + 1,
                    "text_chunk": full_text[start:start + CHUNK_SIZE]
                })

    candidates_out = pd.DataFrame(candidate_rows)
    chunks_out = pd.DataFrame(chunk_rows)
    audit_out = pd.DataFrame(
        audit_rows,
        columns=[
            "record_id", "text_column", "full_text_char_count",
            "excel_cell_limit", "note"
        ]
    )

    print(f"Records: {len(df)}")
    print(f"Text chunks: {len(chunks_out)}")
    print(f"Long-text records: {len(audit_out)}")
    print("Writing workbook...")

    with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
        candidates_out.to_excel(
            writer, index=False, sheet_name="Candidates"
        )
        chunks_out.to_excel(
            writer, index=False, sheet_name="Text_Chunks"
        )
        audit_out.to_excel(
            writer, index=False, sheet_name="Long_Text_Audit"
        )

    print(f"Done: {OUTPUT_XLSX.resolve()}")
    print("The original CSV was not modified.")

if __name__ == "__main__":
    main()
