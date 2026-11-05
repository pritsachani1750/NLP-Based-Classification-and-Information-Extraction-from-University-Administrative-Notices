University of Hyderabad - OCR recovery kit (45 PDFs)

1. Extract this ZIP into the same folder that contains:
   uoh_all500_source_pdfs/
   uoh_all500_download_log.csv

   Example:
   uoh_all500_ssl_fixed_kit/
     run_ocr_45_pdfs.py
     uoh_all500_download_log.csv
     uoh_all500_source_pdfs/
       <notice_id>.pdf

2. Install Python packages:
   python -m pip install pymupdf pytesseract pillow

3. Verify Tesseract:
   tesseract --version
   tesseract --list-langs
   The "eng" language should be listed.

4. Run:
   python run_ocr_45_pdfs.py

Outputs:
   uoh_ocr_45_output/text/<notice_id>.txt
   uoh_ocr_45_results.csv
   uoh_ocr_45_combined.csv

The script processes only the 45 notice IDs flagged in the PDF audit.
It does not alter the original PDFs. It saves a checkpoint after each file
and reuses existing non-empty text files if you run it again.

Note: OCR output is machine-generated and must be reviewed. It is not
automatically ground truth or verified annotation.
