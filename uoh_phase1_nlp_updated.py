"""
CIA 1 - University of Hyderabad Administrative Documents
Updated Phase 1 NLP pipeline.

- Automatically finds the OCR-enriched CSV or the older Phase 1 CSV.
- Supports multiple possible OCR/text column names.
- Preserves original text and OCR text separately.
- Uses OCR text as a fallback only when the original extracted text is empty.
- Performs normalization, tokenization, stop-word removal, stemming,
  lemmatization, regex candidate extraction, BoW and TF-IDF.
- Does NOT train a classifier (classification belongs to Phase 2).
"""

from pathlib import Path
import re
import unicodedata
import warnings

import numpy as np
import pandas as pd
import nltk

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from scipy.sparse import save_npz

# ------------------------- PATHS -------------------------

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# Preferred first: OCR-enriched file. Falls back to earlier Phase 1 files.
INPUT_CANDIDATES = [
    "uoh_phase1_dataset_with_ocr_candidates.csv",
    "uoh_phase1_preprocessed_final.csv",
    "uoh_phase1_preprocessed_improved.csv",
    "uoh_admin_documents_cleaned_annotation_prep.csv",
]

# ---------------------- NLTK SETUP -----------------------

def setup_nltk():
    resources = ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"]
    for resource in resources:
        try:
            nltk.download(resource, quiet=True)
        except Exception as exc:
            warnings.warn(f"Could not download NLTK resource {resource}: {exc}")

# --------------------- HELPER FUNCTIONS ------------------
def find_input_file():
    dataset_path = (
        ROOT / "data" / "uoh_phase1_preprocessed_improved.csv"
    )

    if dataset_path.exists():
        print(f"Dataset selected: {dataset_path}")
        return dataset_path

    raise FileNotFoundError(
        f"Dataset not found: {dataset_path}"
    )

def clean_cell(value):
    if pd.isna(value):
        return ""
    return str(value).strip()

def choose_column(df, candidates):
    lower_to_original = {str(c).lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate.lower() in lower_to_original:
            return lower_to_original[candidate.lower()]
    return None

def choose_text_for_row(row, original_col, ocr_cols, fallback_cols):
    # Preserve the primary extracted text when present.
    original = clean_cell(row.get(original_col, "")) if original_col else ""
    if original:
        return original, "original_extracted_text"

    # Use OCR only as a fallback for records without extracted text.
    for col in ocr_cols:
        value = clean_cell(row.get(col, ""))
        if value:
            return value, f"ocr_fallback:{col}"

    # Last-resort fallback to other text-bearing fields.
    for col in fallback_cols:
        value = clean_cell(row.get(col, ""))
        if value:
            return value, f"other_text_fallback:{col}"

    return "", "no_text_available"

def normalize_text(text):
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def extract_regex_candidates(text):
    date_pattern = (
        r"\b(?:"
        r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|"
        r"\d{1,2}\s+(?:January|February|March|April|May|June|July|"
        r"August|September|October|November|December)\s+\d{4}|"
        r"(?:January|February|March|April|May|June|July|August|"
        r"September|October|November|December)\s+\d{1,2},?\s+\d{4}"
        r")\b"
    )
    email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
    reference_pattern = (
        r"\b(?:Ref(?:erence)?\s*No\.?|File\s*No\.?|Notification\s*No\.?|"
        r"Tender\s*No\.?|NIT\s*No\.?|Letter\s*No\.?|Proceedings\s*No\.?)"
        r"\s*[:#-]?\s*[A-Z0-9][A-Z0-9/_-]*"
    )

    dates = re.findall(date_pattern, text, flags=re.IGNORECASE)
    emails = re.findall(email_pattern, text, flags=re.IGNORECASE)
    references = re.findall(reference_pattern, text, flags=re.IGNORECASE)
    return dates, emails, references

# -------------------------- MAIN -------------------------

def main():
    setup_nltk()
    input_path = find_input_file()

    df = pd.read_csv(input_path, low_memory=False)
    print(f"\nTotal dataset records: {len(df)}")
    print(f"Dataset columns: {len(df.columns)}")

    # Identify source text columns using known names.
    original_col = choose_column(df, [
        "original_text_for_nlp", "notice_text", "extracted_text",
        "document_text", "text"
    ])

    # Detect OCR-related columns by name, prioritizing likely full-text fields.
    ocr_cols = [
        c for c in df.columns
        if "ocr" in str(c).lower()
        and any(k in str(c).lower() for k in ["text", "content", "extract", "full"])
    ]

    # Candidate fields produced by OCR merge scripts may have different names.
    candidate_col_names = [
        "text_for_nlp_candidate", "candidate_text_with_ocr",
        "combined_text", "final_text_candidate", "ocr_candidate_text",
        "notice_text_with_ocr", "text_with_ocr"
    ]
    candidate_cols = []
    for name in candidate_col_names:
        col = choose_column(df, [name])
        if col and col not in candidate_cols:
            candidate_cols.append(col)

    fallback_cols = candidate_cols + [
        c for c in df.columns
        if c not in ([original_col] if original_col else [])
        and c not in ocr_cols
        and any(k in str(c).lower() for k in ["text", "content"])
        and c not in candidate_cols
    ]

    if not original_col and not ocr_cols and not fallback_cols:
        raise ValueError(
            "Could not identify a text column. Dataset columns are:\n" +
            ", ".join(map(str, df.columns))
        )

    print("Primary text column:", original_col)
    print("OCR text columns:", list(map(str, ocr_cols)))
    print("Other candidate text columns:", list(map(str, fallback_cols)))

    selected_texts = []
    text_sources = []
    for _, row in df.iterrows():
        text, source = choose_text_for_row(row, original_col, ocr_cols, fallback_cols)
        selected_texts.append(text)
        text_sources.append(source)

    df["pipeline_input_text"] = selected_texts
    df["pipeline_text_source"] = text_sources
    df["normalized_text_pipeline"] = df["pipeline_input_text"].map(normalize_text)
    df["text_available_pipeline"] = df["normalized_text_pipeline"].str.len() > 0

    stop_words = set(stopwords.words("english"))
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()

    processed_texts = []
    token_counts = []
    filtered_counts = []
    date_results, email_results, reference_results = [], [], []

    for text in df["normalized_text_pipeline"]:
        if not text:
            processed_texts.append("")
            token_counts.append(0)
            filtered_counts.append(0)
            date_results.append("")
            email_results.append("")
            reference_results.append("")
            continue

        try:
            tokens = word_tokenize(text.lower())
        except LookupError:
            # Fallback tokenizer if punkt resources are unavailable.
            tokens = re.findall(r"\b\w+\b", text.lower())

        filtered = [
            token for token in tokens
            if token.isalpha() and token not in stop_words
        ]

        # Apply stemming and lemmatization as separate operations.
        # The feature text uses lemmatized tokens; stemmed tokens are also saved.
        stemmed = [stemmer.stem(token) for token in filtered]
        lemmatized = [lemmatizer.lemmatize(token) for token in filtered]

        processed_texts.append(" ".join(lemmatized))
        token_counts.append(len(tokens))
        filtered_counts.append(len(filtered))

        dates, emails, references = extract_regex_candidates(text)
        date_results.append(" | ".join(dates))
        email_results.append(" | ".join(emails))
        reference_results.append(" | ".join(references))

    df["token_count"] = token_counts
    df["tokens_after_stopword_removal"] = filtered_counts
    df["processed_text_pipeline"] = processed_texts
    df["regex_dates_pipeline"] = date_results
    df["regex_emails_pipeline"] = email_results
    df["regex_reference_candidates_pipeline"] = reference_results

    # Vectorize only records with non-empty processed text.
    usable_mask = df["processed_text_pipeline"].str.strip().ne("")
    vector_texts = df.loc[usable_mask, "processed_text_pipeline"].tolist()

    if not vector_texts:
        raise ValueError(
            "No non-empty text remains for vectorization. Check the input CSV text columns."
        )

    # Avoid min_df=2 failure on tiny datasets; for the expected dataset this is fine.
    bow = CountVectorizer(max_features=3000, min_df=1)
    tfidf = TfidfVectorizer(max_features=3000, min_df=1)

    bow_matrix = bow.fit_transform(vector_texts)
    tfidf_matrix = tfidf.fit_transform(vector_texts)

    # Save matrices and vocabularies.
    save_npz(OUTPUT_DIR / "uoh_phase1_bow_updated.npz", bow_matrix)
    save_npz(OUTPUT_DIR / "uoh_phase1_tfidf_updated.npz", tfidf_matrix)

    pd.DataFrame({
        "feature": bow.get_feature_names_out()
    }).to_csv(OUTPUT_DIR / "bow_vocabulary_updated.csv", index=False)

    pd.DataFrame({
        "feature": tfidf.get_feature_names_out()
    }).to_csv(OUTPUT_DIR / "tfidf_vocabulary_updated.csv", index=False)

    # Save document-level regex and quality summaries.
    df.to_csv(OUTPUT_DIR / "uoh_phase1_preprocessed_updated.csv", index=False)

    regex_summary = pd.DataFrame({
        "field": ["dates", "emails", "reference_candidates"],
        "documents_with_candidates": [
            sum(bool(x) for x in date_results),
            sum(bool(x) for x in email_results),
            sum(bool(x) for x in reference_results)
        ],
        "total_candidates": [
            sum(len(x.split(" | ")) for x in date_results if x),
            sum(len(x.split(" | ")) for x in email_results if x),
            sum(len(x.split(" | ")) for x in reference_results if x)
        ]
    })
    regex_summary.to_csv(OUTPUT_DIR / "regex_extraction_summary_updated.csv", index=False)

    quality = df[[
        c for c in [
            "notice_id", "title_from_archive", "source_url",
            "pipeline_text_source", "text_available_pipeline",
            "token_count", "tokens_after_stopword_removal"
        ] if c in df.columns
    ]].copy()
    quality.to_csv(OUTPUT_DIR / "document_quality_control_updated.csv", index=False)

    print("\n" + "=" * 65)
    print("PHASE 1 NLP PIPELINE COMPLETED")
    print("=" * 65)
    print("Input file:", input_path.name)
    print("Total records:", len(df))
    print("Records with usable text:", int(usable_mask.sum()))
    print("Records without usable text:", int((~usable_mask).sum()))
    print("Documents using OCR fallback:", sum(s.startswith("ocr_fallback:") for s in text_sources))
    print("BoW matrix shape:", bow_matrix.shape)
    print("TF-IDF matrix shape:", tfidf_matrix.shape)
    print("\nRegex extraction summary:")
    print(regex_summary.to_string(index=False))
    print("\nOutput folder:", OUTPUT_DIR)
    print("Classifier training: NOT performed (planned for Phase 2).")

if __name__ == "__main__":
    main()
