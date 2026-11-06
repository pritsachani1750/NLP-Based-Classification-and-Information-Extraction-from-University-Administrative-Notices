from pathlib import Path
import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "linear_svm_tuned_pipeline.joblib"

model = joblib.load(MODEL_PATH)

print("\nUniversity of Hyderabad Notice Classifier")
print("Enter a notice title or full notice text.")
print("Type 'exit' to stop.\n")

while True:
    text = input("Notice text: ").strip()
    if text.lower() in {"exit", "quit"}:
        print("Demo ended.")
        break
    if len(text.split()) < 5:
        print("Please enter a more descriptive notice (at least 5 words).")
        continue

    prediction = model.predict([text])[0]
    print(f"\nPredicted category: {prediction}\n")
