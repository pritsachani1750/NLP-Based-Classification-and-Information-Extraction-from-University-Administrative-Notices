# UoH Phase 1 NLP - Improved VS Code Project

1. Extract this ZIP.
2. Open the extracted folder in VS Code.
3. Run in the terminal:
   `python -m pip install -r requirements.txt`
4. Run:
   `python uoh_phase1_nlp.py`

The master dataset CSV is included. Results are written to `outputs/`.

Improvements:
- Stricter date extraction (numeric and month-name forms).
- Reference extraction requires a meaningful label (e.g. Ref No, File No, Tender No).
- Administrative identifiers are removed from feature text while original source text is preserved.
- Regex candidates remain candidates and require manual source verification.

This remains a Phase 1 preprocessing/vectorization pipeline. No supervised classifier is trained until category labels are human-confirmed.
