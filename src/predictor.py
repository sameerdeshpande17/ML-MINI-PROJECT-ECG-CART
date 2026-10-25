"""
to predict the class of a patient record by taking their row number

project layout
      data/arrhythmia.data
      results/best_model.joblib
      src/train_compare_ecg_models.py
      src/predictor.py

Run from project root directory
    python src/predictor.py --row 0

zero based row numbering i.e. --row 0 selects the patient rocord 1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

DATA_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "arrhythmia/arrhythmia.data"
)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name.lower() == "src" else SCRIPT_DIR
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
DATA_FILE = DATA_DIR / "arrhythmia.data"
MODEL_FILE = RESULTS_DIR / "best_model.joblib"
SUMMARY_FILE = RESULTS_DIR / "run_summary.json"


def load_dataset() -> tuple[pd.DataFrame, pd.Series]:
    """Load the same dataset and apply the same fixed feature removal as training."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    legacy_candidates = [PROJECT_ROOT / "arrhythmia.data", SCRIPT_DIR / "arrhythmia.data"]
    existing_file = next(
        (path for path in [DATA_FILE, *legacy_candidates] if path.exists()),
        None,
    )

    if existing_file is None:
        print("arrhythmia.data was not found. Downloading the UCI dataset...")
        try:
            data = pd.read_csv(DATA_URL, header=None, na_values="?")
        except Exception as exc:
            raise SystemExit(
                "Could not download the UCI dataset. Place arrhythmia.data in "
                f"'{DATA_FILE}' and run again. Details: {exc}"
            ) from exc
        data.to_csv(DATA_FILE, header=False, index=False)
    else:
        data = pd.read_csv(existing_file, header=None, na_values="?")
        if existing_file != DATA_FILE:
            data.to_csv(DATA_FILE, header=False, index=False)

    if data.empty or data.shape[1] < 2:
        raise SystemExit(f"Dataset file is empty or malformed: {DATA_FILE}")

    X = data.iloc[:, :-1].astype(float)
    y = data.iloc[:, -1].astype(int)

    if X.shape[1] >= 14:
        X = X.drop(columns=X.columns[13])

    return X, y


def load_model_name() -> str:
    """Read the selected model name from the run summary when available."""
    if SUMMARY_FILE.exists():
        try:
            summary = json.loads(SUMMARY_FILE.read_text(encoding="utf-8"))
            return str(summary.get("best_model", {}).get("name", "Saved model"))
        except (OSError, json.JSONDecodeError, AttributeError):
            pass
    return "Saved model"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Predict the class for one row from arrhythmia.data using the "
            "previously trained best_model.joblib."
        )
    )
    parser.add_argument(
        "--row",
        type=int,
        required=True,
        help="Zero-based row index (0 is the first patient record).",
    )
    args = parser.parse_args()

    if not MODEL_FILE.exists():
        raise SystemExit(
            "Trained model not found. Expected it at:\n"
            f"  {MODEL_FILE}\n"
            "Run the training script first (for example: python src/new.py)."
        )

    X, y = load_dataset()
    if args.row < 0 or args.row >= len(X):
        raise SystemExit(
            f"Row index {args.row} is out of range. Choose a value from 0 to {len(X) - 1}."
        )

    try:
        model = joblib.load(MODEL_FILE)
    except Exception as exc:
        raise SystemExit(
            f"Could not load the trained model at '{MODEL_FILE}': {exc}"
        ) from exc

    row_features = X.iloc[[args.row]]
    actual_class = int(y.iloc[args.row])
    predicted_class = int(model.predict(row_features)[0])

    print(f"Dataset row:     {args.row} (zero-based)")
    print(f"Model:           {load_model_name()}")
    print(f"Actual class:    {actual_class}")
    print(f"Predicted class: {predicted_class}")
    print(f"Match:           {'Yes' if actual_class == predicted_class else 'No'}")
    print(f"Model file:      {MODEL_FILE}")


if __name__ == "__main__":
    main()
