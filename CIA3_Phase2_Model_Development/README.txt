# CIA 3 Phase 2 – NLP Model Development

## Files
- `data/CIA3_Phase2_Final_Labelled_Dataset.csv`: Phase 2 working dataset.
- `notebooks/CIA3_Phase2_NLP_Model_Development.ipynb`: full model development notebook.
- `models/`: saved model pipelines will be written here when notebook is run.
- `outputs/`: metrics, confusion matrices and error analysis will be written here.

## Run in VS Code (Windows / PowerShell)
1. Extract this project folder.
2. Open the extracted `CIA3_Phase2_Model_Development` folder in VS Code.
3. Open Terminal > New Terminal.
4. Create and activate a virtual environment:
   `py -3.11 -m venv .venv`
   `.\.venv\Scripts\Activate.ps1`
5. Install packages:
   `python -m pip install --upgrade pip`
   `python -m pip install -r requirements.txt`
6. Open `notebooks/CIA3_Phase2_NLP_Model_Development.ipynb`.
7. Select the `.venv` Python kernel and run all cells from top to bottom.

If Python 3.11 is not installed, use a supported Python version that is compatible with the listed packages.

## Important
Labels are rule-assisted provisional labels, not independently human-verified ground truth. The model scores are agreement with these labels. Disclose this limitation in the report and viva. Do not describe the results as verified real-world accuracy.
