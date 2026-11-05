"""
University of Hyderabad Administrative Documents - Final Phase 1 NLP Pipeline
Author: Student project implementation
Purpose: preprocessing, regex information extraction, BoW/TF-IDF, and quality-control outputs.

Important:
- This script does NOT create human-confirmed labels.
- It does NOT train a supervised classifier while verified labels are absent.
- Original document text is preserved; length limits apply only to vectorization input.
"""

from pathlib import Path
import re
import unicodedata
import warnings
import pandas as pd
import numpy as np
from scipy.sparse import save_npz
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import nltk
from nltk.tokenize import wordpunct_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "uoh_phase1_preprocessed_improved.csv"
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

MAX_FEATURES = 3000
MAX_VECTORIZATION_CHARS = 100_000  # only affects feature matrices, never source text
MIN_DF = 2

def ensure_nltk():
    resources = [
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
        ("corpora/omw-1.4", "omw-1.4"),
    ]
    for path, package in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            print(f"Downloading NLTK resource: {package}")
            nltk.download(package, quiet=False)

def normalize_text(value):
    if pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKC", str(value))
    text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F]", " ", text)
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text

DATE_PATTERNS = [
    re.compile(r"\b(?:0?[1-9]|[12]\d|3[01])[-/.](?:0?[1-9]|1[0-2])[-/.](?:19|20)\d{2}\b"),
    re.compile(r"\b(?:19|20)\d{2}[-/.](?:0?[1-9]|1[0-2])[-/.](?:0?[1-9]|[12]\d|3[01])\b"),
    re.compile(r"\b(?:0?[1-9]|[12]\d|3[01])\s+(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+(?:19|20)\d{2}\b",
               re.IGNORECASE),
    re.compile(r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+(?:0?[1-9]|[12]\d|3[01]),?\s+(?:19|20)\d{2}\b",
               re.IGNORECASE),
]
EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
REF_RE = re.compile(
    r"\b(?:Ref(?:erence)?\.?\s*(?:No\.?|Number)?|File\s*No\.?|"
    r"Notification\s*No\.?|Tender\s*No\.?|NIT\s*No\.?|"
    r"Letter\s*No\.?|Proceedings\s*No\.?)\s*[:#\-]?\s*"
    r"([A-Z0-9][A-Z0-9./()_\-]{2,}(?:\s*/\s*[A-Z0-9][A-Z0-9./()_\-]*)*)",
    re.IGNORECASE
)

def unique_matches(patterns, text):
    found = []
    for pattern in patterns:
        found.extend(m.group(0).strip() for m in pattern.finditer(text))
    return list(dict.fromkeys(found))

def extract_fields(text):
    dates = unique_matches(DATE_PATTERNS, text)
    emails = list(dict.fromkeys(EMAIL_RE.findall(text)))
    refs = []
    for match in REF_RE.finditer(text):
        candidate = re.sub(r"\s+", "", match.group(1)).strip(".,;:")
        # Reject obvious OCR/word fragments and candidates without a digit.
        if len(candidate) >= 4 and re.search(r"\d", candidate):
            refs.append(candidate)
    return dates, emails, list(dict.fromkeys(refs))

def preprocess_for_features(text, stop_words, stemmer, lemmatizer):
    tokens = wordpunct_tokenize(text.lower())
    output = []
    for token in tokens:
        if not token.isalpha() or token in stop_words or len(token) < 2:
            continue
        lemma = lemmatizer.lemmatize(token)
        output.append(stemmer.stem(lemma))
    return " ".join(output)

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Place uoh_phase1_preprocessed_improved.csv beside this script."
        )

    ensure_nltk()
    stop_words = set(stopwords.words("english"))
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()

    df = pd.read_csv(DATA_PATH)
    text_col = "notice_text" if "notice_text" in df.columns else "original_text_for_nlp"
    if text_col not in df.columns:
        raise ValueError("Expected notice_text or original_text_for_nlp column in dataset.")

    df["original_text_preserved"] = df[text_col].fillna("").astype(str)
    df["normalized_text_final"] = df["original_text_preserved"].map(normalize_text)
    df["text_available_final"] = df["normalized_text_final"].str.len().gt(0)

    date_lists, email_lists, ref_lists = [], [], []
    processed = []
    vector_input = []
    for text in df["normalized_text_final"]:
        dates, emails, refs = extract_fields(text)
        date_lists.append(" | ".join(dates))
        email_lists.append(" | ".join(emails))
        ref_lists.append(" | ".join(refs))
        # Preserve full normalized text in CSV; cap only the vectorizer input.
        feature_source = text[:MAX_VECTORIZATION_CHARS]
        processed_text = preprocess_for_features(
            feature_source, stop_words, stemmer, lemmatizer
        ) if feature_source else ""
        processed.append(processed_text)
        vector_input.append(processed_text)

    df["date_candidates_final"] = date_lists
    df["email_candidates_final"] = email_lists
    df["reference_candidates_final"] = ref_lists
    df["processed_text_final"] = processed
    df["vectorization_truncated"] = df["normalized_text_final"].str.len().gt(MAX_VECTORIZATION_CHARS)

    usable = df.loc[df["processed_text_final"].str.strip().ne("")].copy()
    if usable.empty:
        raise ValueError("No usable text remains after preprocessing.")

    bow = CountVectorizer(max_features=MAX_FEATURES, min_df=MIN_DF)
    tfidf = TfidfVectorizer(max_features=MAX_FEATURES, min_df=MIN_DF)
    bow_matrix = bow.fit_transform(usable["processed_text_final"])
    tfidf_matrix = tfidf.fit_transform(usable["processed_text_final"])

    save_npz(OUTPUT_DIR / "uoh_phase1_bow_final.npz", bow_matrix)
    save_npz(OUTPUT_DIR / "uoh_phase1_tfidf_final.npz", tfidf_matrix)
    pd.DataFrame({"term": bow.get_feature_names_out()}).to_csv(
        OUTPUT_DIR / "bow_vocabulary_final.csv", index=False
    )
    pd.DataFrame({"term": tfidf.get_feature_names_out()}).to_csv(
        OUTPUT_DIR / "tfidf_vocabulary_final.csv", index=False
    )
    df.to_csv(OUTPUT_DIR / "uoh_phase1_preprocessed_final.csv", index=False)

    summary = []
    for field, label in [
        ("date_candidates_final", "date"),
        ("email_candidates_final", "email"),
        ("reference_candidates_final", "reference"),
    ]:
        counts = df[field].fillna("").map(
            lambda x: len([v for v in x.split(" | ") if v.strip()])
        )
        summary.append({
            "field": label,
            "documents_with_candidates": int(counts.gt(0).sum()),
            "total_candidates": int(counts.sum()),
        })
    pd.DataFrame(summary).to_csv(
        OUTPUT_DIR / "regex_extraction_summary_final.csv", index=False
    )

    # Quality-control report; these are flags for manual inspection, not errors by definition.
    quality = pd.DataFrame({
        "notice_id": df.get("notice_id", pd.Series(range(1, len(df) + 1))),
        "title": df.get("title_from_archive", pd.Series([""] * len(df))),
        "source_url": df.get("source_url", pd.Series([""] * len(df))),
        "text_characters": df["normalized_text_final"].str.len(),
        "text_empty": ~df["text_available_final"],
        "very_long_text_over_100k": df["normalized_text_final"].str.len().gt(100_000),
        "vectorization_truncated": df["vectorization_truncated"],
        "dates_found": df["date_candidates_final"].map(
            lambda x: len([v for v in x.split(" | ") if v.strip()])
        ),
        "emails_found": df["email_candidates_final"].map(
            lambda x: len([v for v in x.split(" | ") if v.strip()])
        ),
        "references_found": df["reference_candidates_final"].map(
            lambda x: len([v for v in x.split(" | ") if v.strip()])
        ),
    })
    quality.to_csv(OUTPUT_DIR / "document_quality_control_final.csv", index=False)

    labels_col = "final_category_human"
    confirmed = 0
    if labels_col in df.columns:
        confirmed = int(df[labels_col].fillna("").astype(str).str.strip().ne("").sum())

    print("\n========== FINAL PHASE 1 SUMMARY ==========")
    print(f"Total records: {len(df)}")
    print(f"Documents with extracted text: {int(df['text_available_final'].sum())}")
    print(f"Documents with empty text: {int((~df['text_available_final']).sum())}")
    print(f"Documents with human-confirmed labels: {confirmed}")
    print(f"Documents used for vectorization: {len(usable)}")
    print(f"Documents capped at {MAX_VECTORIZATION_CHARS:,} chars for vectorization: {int(df['vectorization_truncated'].sum())}")
    print(f"BoW matrix shape: {bow_matrix.shape}")
    print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
    print("\nRegex extraction:")
    for row in summary:
        print(f"{row['field'].title():10s}: {row['documents_with_candidates']} documents, {row['total_candidates']} candidates")
    print(f"\nOutputs saved to: {OUTPUT_DIR}")
    print("\nIMPORTANT: Regex candidates require checking against original PDFs.")
    print("No supervised classifier is trained unless human-confirmed labels are available.")

if __name__ == "__main__":
    main()
