# University of Hyderabad Administrative Documents — Final Phase 1 NLP

## Files
- `uoh_phase1_nlp_final.py`: complete preprocessing, regex extraction, BoW, TF-IDF and QC pipeline.
- `uoh_phase1_preprocessed_improved.csv`: input dataset (500 records).
- `requirements.txt`: Python dependencies.
- `outputs/`: generated when the script runs.

## Run in VS Code (Windows PowerShell)

1. Extract this ZIP using **Extract All**.
2. Open the extracted folder in VS Code: **File → Open Folder**.
3. Open the VS Code terminal (**Terminal → New Terminal**).
4. Create and activate a virtual environment (recommended):

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

5. Install packages:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

6. Run:

```powershell
python uoh_phase1_nlp_final.py
```

The first run may download NLTK resources. Allow it to finish.

## Generated outputs
- `outputs/uoh_phase1_preprocessed_final.csv`
- `outputs/uoh_phase1_bow_final.npz`
- `outputs/uoh_phase1_tfidf_final.npz`
- `outputs/bow_vocabulary_final.csv`
- `outputs/tfidf_vocabulary_final.csv`
- `outputs/regex_extraction_summary_final.csv`
- `outputs/document_quality_control_final.csv`

## Important academic notes
- The original text is preserved. The 100,000-character cap applies only to vectorization input, not to the saved source text.
- Date, email and reference-number patterns are candidate extraction results and must be manually validated against source documents before reporting extraction accuracy.
- Empty-text records are retained and flagged for PDF/OCR/manual review.
- Category suggestions are not ground truth. A supervised classifier must not be evaluated or reported until the human-confirmed labels are completed.
- This is an English-focused baseline. Hindi or other-language documents need language-aware preprocessing or separate handling.
