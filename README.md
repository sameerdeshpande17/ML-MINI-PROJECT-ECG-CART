# ECG Arrhythmia Classification

**Machine Learning Mini-Project — UE24CS352A**  
**Team:** Sameer Deshpande and Supreeth Sharma Y V

## About the project

We use the UCI Arrhythmia dataset to classify ECG records. CART is our baseline, and we compare it with balanced CART, Random Forest, Extra Trees, Gradient Boosting and balanced Logistic Regression.

The dataset has 452 records, 278 input features after removing feature 14, and 13 observed classes. We use a stratified 70/30 train-test split. Missing values are filled using median imputation inside each model's pipeline.

## Run the project

Use Python 3.10 or later. From the project folder, run:

```bash
pip install -r requirements.txt
python src/train_compare_ecg_models.py
```

The first run downloads `arrhythmia.data` into `data/` if it is not already there. The script trains and compares the six models, calculates multiclass ROC-AUC using One-vs-Rest, and saves the results in `results/`.

## Predict a dataset row

After training, run:

```bash
python src/predictor.py --row 0
```

Change `0` to another zero-based row number. The predictor shows the row's actual class and the model's predicted class. Since the data row already has a label, this is a demonstration using a known record, not a diagnosis for a new patient.

## Main results

In our current run, Gradient Boosting performed best by Macro F1 (0.4036) and accuracy (73.53%). Random Forest had the highest macro ROC-AUC (0.8775) and weighted ROC-AUC (0.8890). These results come from one test split, so they should be treated as preliminary.

## Files created in `results/`

- `best_model.joblib` — saved winning model and preprocessing steps
- `model_comparison.csv` — comparison of all six models
- `roc_auc_comparison.csv` and `multiclass_roc_curves.png` — ROC-AUC scores and plot
- `best_model_classification_report.csv` and `.txt` — per-class performance
- `best_model_confusion_matrix.csv` — actual versus predicted class counts
- `test_set_predictions.csv` — predictions from all six models
- `run_summary.json` and `run_summary.txt` — details of the run

## Note

The dataset is strongly imbalanced, and some classes have very few records. Their scores can be unstable. The best model was selected using Macro F1 on the same holdout set used for reporting results, so its score may be optimistic. This project is for learning and is not a clinical diagnostic tool.

## Reference

UCI Machine Learning Repository, [Arrhythmia dataset](https://doi.org/10.24432/C5BS32).
