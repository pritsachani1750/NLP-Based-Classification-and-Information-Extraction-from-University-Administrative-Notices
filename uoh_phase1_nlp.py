"""
CIA 1 Phase 1 - Improved NLP pipeline
NLP-Based Classification and Information Extraction from University Administrative Documents

Run:
    python uoh_phase1_nlp.py

Keep the master CSV in this folder:
    uoh_admin_documents_cleaned_annotation_prep.csv

This script performs preprocessing, candidate information extraction, BoW and TF-IDF.
It does not train a supervised classifier because verified labels are not yet available.
"""
from pathlib import Path
import re
import unicodedata
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "uoh_admin_documents_cleaned_annotation_prep.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# Stricter numeric date: DD/MM/YYYY, DD-MM-YYYY, DD.MM.YYYY
NUMERIC_DATE = re.compile(
    r"(?<!\d)(0?[1-9]|[12]\d|3[01])([/.-])(0?[1-9]|1[0-2])\2((?:19|20)\d{2})(?!\d)"
)
# Month-name dates such as 12 March 2025 or March 12, 2025
MONTH_DATE = re.compile(
    r"\b(?:0?[1-9]|[12]\d|3[01])\s+"
    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
    r"\s*,?\s*(?:19|20)\d{2}\b",
    re.IGNORECASE
)
MONTH_DATE_REVERSED = re.compile(
    r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
    r"\s+[0-3]?\d\s*,?\s*(?:19|20)\d{2}\b",
    re.IGNORECASE
)
EMAIL_PATTERN = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
)
# Only capture identifiers when preceded by a meaningful reference label.
REFERENCE_PATTERN = re.compile(
    r"\b(?:Reference\s*(?:No\.?|Number)?|Ref\.?\s*(?:No\.?|Number)?|"
    r"File\s*(?:No\.?|Number)|Notification\s*(?:No\.?|Number)|"
    r"Tender\s*(?:No\.?|Number)|NIT\s*(?:No\.?|Number)|"
    r"Letter\s*(?:No\.?|Number)|Proceedings\s*(?:No\.?|Number))"
    r"\s*[:#-]?\s*([A-Z0-9][A-Z0-9./_-]{2,})",
    re.IGNORECASE
)
# Remove obvious administrative identifiers from feature text, not from source text.
IDENTIFIER_CLEANUP = re.compile(
    r"\b(?:UOH|UoH|NIT|GEM|F\.No|File\s*No)\s*[/.:#-]?\s*"
    r"[A-Z0-9][A-Z0-9./_-]{2,}\b",
    re.IGNORECASE
)

def normalize_text(value):
    value = unicodedata.normalize("NFKC", str(value))
    value = value.lower()
    value = re.sub(r"[\t\r\n]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()

def extract_dates(text):
    found = set()
    for match in NUMERIC_DATE.finditer(str(text)):
        found.add(match.group(0))
    for pattern in (MONTH_DATE, MONTH_DATE_REVERSED):
        found.update(m.group(0) for m in pattern.finditer(str(text)))
    return sorted(found)

def extract_emails(text):
    return sorted(set(EMAIL_PATTERN.findall(str(text))))

def extract_references(text):
    # Group 1 contains the identifier, excluding the generic label itself.
    return sorted(set(m.group(1).rstrip(".,;:") for m in REFERENCE_PATTERN.finditer(str(text))))

def clean_for_features(text):
    text = normalize_text(text)
    text = IDENTIFIER_CLEANUP.sub(" ", text)
    text = re.sub(r"\b(?:ref|reference|file\s*no|notification\s*no|tender\s*no|nit\s*no)\b", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Copy uoh_admin_documents_cleaned_annotation_prep.csv into this folder."
        )

    df = pd.read_csv(DATA_PATH).fillna("")
    text_col = "cleaned_text" if "cleaned_text" in df.columns else "notice_text"
    title_col = "cleaned_title" if "cleaned_title" in df.columns else "title_from_archive"

    df["original_text_for_nlp"] = df[text_col].astype(str)
    df["text_available"] = df["original_text_for_nlp"].str.strip().ne("")
    df["normalized_text"] = df["original_text_for_nlp"].apply(normalize_text)

    print("\n=== DATASET SUMMARY ===")
    print("Total records:", len(df))
    print("Records with extracted text:", int(df["text_available"].sum()))
    print("Records with empty extracted text:", int((~df["text_available"]).sum()))
    if "final_category_human" in df.columns:
        print("Human-confirmed labels:",
              int(df["final_category_human"].astype(str).str.strip().ne("").sum()))

    # Candidate extraction; these are not ground-truth facts until checked.
    df["regex_dates"] = df["original_text_for_nlp"].apply(lambda x: " | ".join(extract_dates(x)))
    df["regex_emails"] = df["original_text_for_nlp"].apply(lambda x: " | ".join(extract_emails(x)))
    df["regex_reference_candidates"] = df["original_text_for_nlp"].apply(
        lambda x: " | ".join(extract_references(x))
    )

    # English NLTK demonstration. Non-English text needs language-specific processing.
    import nltk
    for resource in ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"]:
        try:
            nltk.download(resource, quiet=True)
        except Exception as exc:
            print(f"NLTK download warning for {resource}: {exc}")

    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer, WordNetLemmatizer

    try:
        stop_words = set(stopwords.words("english"))
    except LookupError:
        stop_words = set()

    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()

    def tokenize_english(text):
        tokens = word_tokenize(clean_for_features(text))
        # Keep alphabetic English tokens only; source and extracted fields remain preserved.
        return [token for token in tokens if token.isalpha() and token.isascii()]

    def preprocess(text):
        tokens = tokenize_english(text)
        tokens = [token for token in tokens if token.lower() not in stop_words]
        return [token.lower() for token in tokens if len(token) > 1]

    sample = next((x for x in df["original_text_for_nlp"] if x.strip()),
                  "University administrative notification")
    sample_tokens = preprocess(sample)
    print("\n=== IMPROVED PREPROCESSING EXAMPLE ===")
    print("Tokens:", sample_tokens[:30])
    print("Stemming:", [stemmer.stem(t) for t in sample_tokens[:30]])
    print("Lemmatization:", [lemmatizer.lemmatize(t) for t in sample_tokens[:30]])

    df["processed_tokens"] = df["original_text_for_nlp"].apply(preprocess)
    df["processed_text"] = df["processed_tokens"].apply(lambda items: " ".join(items))

    usable = df["processed_text"].str.strip().ne("")
    corpus = df.loc[usable, "processed_text"]
    if corpus.empty:
        raise ValueError("No non-empty documents available for vectorization.")

    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    from scipy.sparse import save_npz

    bow_vectorizer = CountVectorizer(max_features=3000, ngram_range=(1, 1))
    X_bow = bow_vectorizer.fit_transform(corpus)
    tfidf_vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 2))
    X_tfidf = tfidf_vectorizer.fit_transform(corpus)

    out_df = df.drop(columns=["processed_tokens"]).copy()
    out_df.to_csv(OUTPUT_DIR / "uoh_phase1_preprocessed_improved.csv",
                  index=False, encoding="utf-8-sig")
    save_npz(OUTPUT_DIR / "uoh_phase1_bow_improved.npz", X_bow)
    save_npz(OUTPUT_DIR / "uoh_phase1_tfidf_improved.npz", X_tfidf)
    pd.DataFrame({"term": bow_vectorizer.get_feature_names_out()}).to_csv(
        OUTPUT_DIR / "bow_vocabulary_improved.csv", index=False)
    pd.DataFrame({"term": tfidf_vectorizer.get_feature_names_out()}).to_csv(
        OUTPUT_DIR / "tfidf_vocabulary_improved.csv", index=False)

    regex_summary = pd.DataFrame({
        "Metric": ["Documents with date candidates", "Total date candidates",
                   "Documents with email candidates", "Total email candidates",
                   "Documents with reference candidates", "Total reference candidates"],
        "Count": [
            int(df["regex_dates"].str.strip().ne("").sum()),
            int(df["regex_dates"].apply(lambda x: len(x.split(" | ")) if x else 0).sum()),
            int(df["regex_emails"].str.strip().ne("").sum()),
            int(df["regex_emails"].apply(lambda x: len(x.split(" | ")) if x else 0).sum()),
            int(df["regex_reference_candidates"].str.strip().ne("").sum()),
            int(df["regex_reference_candidates"].apply(lambda x: len(x.split(" | ")) if x else 0).sum())
        ]
    })
    regex_summary.to_csv(OUTPUT_DIR / "regex_extraction_summary.csv", index=False)

    print("\n=== IMPROVED VECTOR REPRESENTATIONS ===")
    print("Documents vectorized:", len(corpus))
    print("BoW matrix shape:", X_bow.shape)
    print("TF-IDF matrix shape:", X_tfidf.shape)
    print("\n=== REGEX CANDIDATE SUMMARY ===")
    print(regex_summary.to_string(index=False))
    print("\nOutputs saved in:", OUTPUT_DIR)
    print("Validate extracted candidates against original PDFs before reporting accuracy.")
    print("No supervised classifier trained: human-confirmed labels are still pending.")

if __name__ == "__main__":
    main()
