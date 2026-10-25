"""
ECG arrhythmia classification, comparing CART baseline vs ensemble models

we have used one stratified 70/30 hold-out split. 
Preprocessing is fitted only on training data through sklearn pipelines so that there is no Data leakage
All observed classes are retained.

outputs are written to /results/ and the dataset cache is stored in /data/.
"""

from pathlib import Path
import json
import re

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

RANDOM_STATE = 42
DATA_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "arrhythmia/arrhythmia.data"
)

# Resolve paths from this script, not from the terminal's current directory.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name.lower() == "src" else SCRIPT_DIR
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
DATA_FILE = DATA_DIR / "arrhythmia.data"


def load_data():
    """Load the cached UCI data, or download and cache it under data/."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Also accept a cache created by an earlier version of the script.
    legacy_candidates = [PROJECT_ROOT / "arrhythmia.data", SCRIPT_DIR / "arrhythmia.data"]
    cached_file = next(
        (path for path in [DATA_FILE, *legacy_candidates] if path.exists()),
        None,
    )

    if cached_file is not None:
        data = pd.read_csv(cached_file, header=None, na_values="?")
        if cached_file != DATA_FILE:
            data.to_csv(DATA_FILE, header=False, index=False)
    else:
        print("Downloading UCI Arrhythmia dataset ...", flush=True)
        data = pd.read_csv(DATA_URL, header=None, na_values="?")
        data.to_csv(DATA_FILE, header=False, index=False)

    X = data.iloc[:, :-1].astype(float)
    y = data.iloc[:, -1].astype(int)

    # Dataset feature 14 is mostly missing; remove it as in the earlier work.
    # This is a fixed, dataset-specific choice, not a data-driven test-set step.
    if X.shape[1] >= 14:
        X = X.drop(columns=X.columns[13])

    return X, y


def make_pipeline(model, scale=False):
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", model))
    return Pipeline(steps)


def safe_name(name):
    """Convert a model label into a filename-friendly identifier."""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def aligned_predict_proba(model, X, labels):
    """Return predicted probabilities with columns ordered by the given labels."""
    raw_probabilities = model.predict_proba(X)
    fitted_classes = model.named_steps["model"].classes_
    fitted_index = {label: index for index, label in enumerate(fitted_classes)}
    aligned = np.zeros((len(X), len(labels)), dtype=float)
    for column, label in enumerate(labels):
        if label in fitted_index:
            aligned[:, column] = raw_probabilities[:, fitted_index[label]]
    return aligned


def macro_roc_curve(y_true, probabilities, labels):
    """Build a macro-averaged one-vs-rest ROC curve for multiclass labels."""
    y_array = np.asarray(y_true)
    mean_fpr = np.linspace(0.0, 1.0, 201)
    class_tprs = []

    for column, label in enumerate(labels):
        binary_true = (y_array == label).astype(int)
        # An ROC curve needs both positive and negative examples for this class.
        if np.unique(binary_true).size < 2:
            continue
        fpr, tpr, _ = roc_curve(binary_true, probabilities[:, column])
        class_tprs.append(np.interp(mean_fpr, fpr, tpr))

    if not class_tprs:
        raise ValueError("Cannot calculate a multiclass ROC curve: no class has both positive and negative test examples.")

    mean_tpr = np.mean(class_tprs, axis=0)
    mean_tpr[0] = 0.0
    mean_tpr[-1] = 1.0
    return mean_fpr, mean_tpr


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    X, y = load_data()
    labels = sorted(y.unique())
    class_counts = y.value_counts().sort_index()
    print(f"Dataset: {len(X)} rows, {X.shape[1]} features, {y.nunique()} classes")
    print("Class counts:")
    print(class_counts.to_string())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=RANDOM_STATE, stratify=y
    )
    print(f"\nSplit: {len(X_train)} training rows, {len(X_test)} test rows")

    models = {
        "CART (Gini)": make_pipeline(
            DecisionTreeClassifier(random_state=RANDOM_STATE)
        ),
        "CART (balanced)": make_pipeline(
            DecisionTreeClassifier(
                class_weight="balanced", random_state=RANDOM_STATE
            )
        ),
        "Random Forest": make_pipeline(
            RandomForestClassifier(
                n_estimators=300,
                class_weight="balanced_subsample",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
        ),
        "Extra Trees": make_pipeline(
            ExtraTreesClassifier(
                n_estimators=300,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
        ),
        "Gradient Boosting": make_pipeline(
            GradientBoostingClassifier(random_state=RANDOM_STATE)
        ),
        "Logistic Regression (balanced)": make_pipeline(
            LogisticRegression(
                max_iter=3000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
            scale=True,
        ),
    }

    rows = []
    predictions = {}
    roc_details = {}
    for name, model in models.items():
        print(f"Training {name} ...", flush=True)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        probabilities = aligned_predict_proba(model, X_test, labels)
        predictions[name] = pred

        # Multiclass ROC-AUC uses a one-vs-rest (OvR) strategy. Macro AUC
        # weights classes equally; weighted AUC weights classes by support.
        try:
            auc_macro = roc_auc_score(
                y_test, probabilities, labels=labels,
                multi_class="ovr", average="macro"
            )
            auc_weighted = roc_auc_score(
                y_test, probabilities, labels=labels,
                multi_class="ovr", average="weighted"
            )
        except ValueError as exc:
            # Keep the rest of the model comparison if a future split is missing
            # a class needed for multiclass ROC-AUC. NaN makes this explicit.
            print(f"Warning: could not calculate multiclass ROC-AUC for {name}: {exc}")
            auc_macro = np.nan
            auc_weighted = np.nan

        try:
            roc_details[name] = macro_roc_curve(y_test, probabilities, labels)
        except ValueError as exc:
            print(f"Warning: could not draw the multiclass ROC curve for {name}: {exc}")

        rows.append(
            {
                "Model": name,
                "Accuracy": accuracy_score(y_test, pred),
                "Balanced accuracy": balanced_accuracy_score(y_test, pred),
                "Macro F1": f1_score(
                    y_test, pred, average="macro", zero_division=0
                ),
                "Weighted F1": f1_score(
                    y_test, pred, average="weighted", zero_division=0
                ),
                "ROC AUC (macro OvR)": auc_macro,
                "ROC AUC (weighted OvR)": auc_weighted,
            }
        )

    results = pd.DataFrame(rows).sort_values(
        "Macro F1", ascending=False
    ).reset_index(drop=True)
    print("\nTest-set results (sorted by Macro F1):")
    print(
        results.to_string(
            index=False,
            formatters={
                "Accuracy": "{:.4f}".format,
                "Balanced accuracy": "{:.4f}".format,
                "Macro F1": "{:.4f}".format,
                "Weighted F1": "{:.4f}".format,
                "ROC AUC (macro OvR)": "{:.4f}".format,
                "ROC AUC (weighted OvR)": "{:.4f}".format,
            },
        )
    )

    # Save the full model comparison as numeric CSV values.
    results_path = RESULTS_DIR / "model_comparison.csv"
    results.to_csv(results_path, index=False, float_format="%.6f")

    # Save a focused AUC table for the report/presentation.
    auc_results_path = RESULTS_DIR / "roc_auc_comparison.csv"
    results[["Model", "ROC AUC (macro OvR)", "ROC AUC (weighted OvR)"]].sort_values(
        "ROC AUC (macro OvR)", ascending=False, na_position="last"
    ).to_csv(auc_results_path, index=False, float_format="%.6f")

    # Plot macro-averaged one-vs-rest ROC curves for a visual model comparison.
    roc_plot_path = RESULTS_DIR / "multiclass_roc_curves.png"
    fig, ax = plt.subplots(figsize=(9, 7))
    for name in models:
        if name not in roc_details:
            continue
        mean_fpr, mean_tpr = roc_details[name]
        auc_value = results.loc[results["Model"] == name, "ROC AUC (macro OvR)"].iloc[0]
        ax.plot(mean_fpr, mean_tpr, linewidth=1.8, label=f"{name} (AUC={auc_value:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", linewidth=1, label="Chance")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Multiclass ROC Curves (Macro One-vs-Rest)")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(roc_plot_path, dpi=180)
    plt.close(fig)

    best_name = results.iloc[0]["Model"]
    best_model = models[best_name]
    best_prediction = predictions[best_name]
    print(f"\nBest model by hold-out Macro F1: {best_name}")
    print("Per-class report for best model:")
    report_text = classification_report(
        y_test, best_prediction, labels=labels, zero_division=0
    )
    print(report_text)

    # Save the exact fitted pipeline used to generate the selected model's
    # hold-out predictions. This includes the imputer/scaler and classifier.
    model_path = RESULTS_DIR / "best_model.joblib"
    joblib.dump(best_model, model_path)

    # Machine-readable and human-readable per-class classification reports.
    report_dict = classification_report(
        y_test,
        best_prediction,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )
    report_csv = RESULTS_DIR / "best_model_classification_report.csv"
    pd.DataFrame(report_dict).transpose().to_csv(report_csv, float_format="%.6f")
    report_txt = RESULTS_DIR / "best_model_classification_report.txt"
    report_txt.write_text(report_text, encoding="utf-8")

    # Confusion matrix, with named actual/predicted classes in the CSV.
    cm = confusion_matrix(y_test, best_prediction, labels=labels)
    cm_df = pd.DataFrame(
        cm,
        index=[f"actual_{label}" for label in labels],
        columns=[f"predicted_{label}" for label in labels],
    )
    cm_path = RESULTS_DIR / "best_model_confusion_matrix.csv"
    cm_df.to_csv(cm_path, index_label="actual_class")
    print("Confusion matrix (rows=true, columns=predicted):")
    print("Labels:", labels)
    print(cm)

    # Save true labels and predictions for every evaluated model on the same test set.
    prediction_data = {
        "dataset_row_index": y_test.index,
        "true_label": y_test.to_numpy(),
    }
    for name, pred in predictions.items():
        prediction_data[f"pred_{safe_name(name)}"] = pred
    predictions_path = RESULTS_DIR / "test_set_predictions.csv"
    pd.DataFrame(prediction_data).to_csv(predictions_path, index=False)

    best_metrics = results.iloc[0].to_dict()
    summary = {
        "dataset": {
            "source": "UCI Arrhythmia dataset",
            "rows": int(len(X)),
            "features_after_fixed_feature_drop": int(X.shape[1]),
            "number_of_observed_classes": int(y.nunique()),
            "class_counts": {str(int(k)): int(v) for k, v in class_counts.items()},
        },
        "evaluation": {
            "method": "stratified 70/30 train-test split",
            "random_state": RANDOM_STATE,
            "training_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "selection_metric": "Macro F1 on this hold-out split",
            "roc_auc_method": "Multiclass one-vs-rest (OvR); macro and weighted averages",
            "caution": (
                "The best model was selected using hold-out test Macro F1; this can "
                "make the reported winning score optimistic. Rare-class metrics "
                "are unstable because some classes have very few examples."
            ),
        },
        "best_model": {
            "name": best_name,
            "file": model_path.name,
            "metrics": {k: float(v) for k, v in best_metrics.items() if k != "Model"},
            "saved_pipeline_is_fitted_on": "training split only",
            "input_features_in_order": [str(c) for c in X.columns],
            "target_labels": [int(label) for label in labels],
        },
        "files": {
            "model_comparison": results_path.name,
            "roc_auc_comparison": auc_results_path.name,
            "multiclass_roc_curves_plot": roc_plot_path.name,
            "best_model_classification_report_csv": report_csv.name,
            "best_model_classification_report_txt": report_txt.name,
            "best_model_confusion_matrix": cm_path.name,
            "test_set_predictions": predictions_path.name,
        },
    }
    summary_path = RESULTS_DIR / "run_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Compact run summary that is easy to read or include in a report.
    summary_text_path = RESULTS_DIR / "run_summary.txt"
    summary_text_path.write_text(
        "ECG Arrhythmia Classification - Run Summary\n"
        "===========================================\n"
        f"Dataset rows: {len(X)}\n"
        f"Features used: {X.shape[1]}\n"
        f"Observed classes: {y.nunique()}\n"
        f"Train/test rows: {len(X_train)}/{len(X_test)}\n"
        f"Best model by hold-out Macro F1: {best_name}\n"
        f"Accuracy: {best_metrics['Accuracy']:.4f}\n"
        f"Balanced accuracy: {best_metrics['Balanced accuracy']:.4f}\n"
        f"Macro F1: {best_metrics['Macro F1']:.4f}\n"
        f"Weighted F1: {best_metrics['Weighted F1']:.4f}\n"
        f"ROC AUC (macro OvR): {best_metrics['ROC AUC (macro OvR)']:.4f}\n"
        f"ROC AUC (weighted OvR): {best_metrics['ROC AUC (weighted OvR)']:.4f}\n\n"
        "Multiclass ROC-AUC uses a one-vs-rest strategy. Macro AUC weights classes equally;\n"
        "weighted AUC weights classes by support.\n"
        "ROC-AUC is probability-based and may also be unstable for rare classes.\n"
        "Note: the selected model was chosen using hold-out Macro F1, so its score\n"
        "may be optimistic. Rare-class metrics are unstable with very few examples.\n",
        encoding="utf-8",
    )

    print("\nSaved outputs to:", RESULTS_DIR)
    print(f"  Trained model:       {model_path}")
    print(f"  Model comparison:    {results_path}")
    print(f"  ROC-AUC comparison:  {auc_results_path}")
    print(f"  ROC curve plot:      {roc_plot_path}")
    print(f"  Classification CSV:  {report_csv}")
    print(f"  Classification TXT:  {report_txt}")
    print(f"  Confusion matrix:    {cm_path}")
    print(f"  Test predictions:    {predictions_path}")
    print(f"  Run summary JSON:    {summary_path}")
    print(f"  Run summary TXT:     {summary_text_path}")
    print("\nNote: rare classes have very few examples, so their test metrics are unstable.")
    print(
        "Compare these models on this same split; do not treat this as a direct "
        "paper comparison unless its protocol matches."
    )


if __name__ == "__main__":
    main()
