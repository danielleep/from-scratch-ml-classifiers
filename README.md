# From-Scratch ML Classifiers

A machine learning project focused on implementing, evaluating, and validating Logistic Regression from scratch using Python and NumPy.

The project is organized as a reproducible experiment pipeline, with separate modules for data loading, preprocessing, modeling, validation, metrics, and visualization.

## Dataset

This project uses the Breast Cancer Wisconsin dataset from `scikit-learn`.

- 569 samples
- 30 numeric features
- Binary classification: malignant / benign

## Implemented Components

- From-scratch Logistic Regression in NumPy, including sigmoid prediction, binary cross-entropy loss, and batch gradient descent
- Reproducible training and evaluation pipeline with train/test splitting, feature standardization, and bias handling
- Custom evaluation metrics and visualizations, including precision, recall, F1 score, ROC-AUC, confusion matrix, and training loss curve
- Learning-rate comparison, 5-fold cross-validation, and scikit-learn LogisticRegression baseline comparison
- Saved experiment outputs, metrics, and figures under `results/`

## Project Structure

```text
src/
├── data.py
├── preprocessing.py
├── logistic_regression.py
├── metrics.py
├── validation.py
└── visualization.py

experiments/
├── run_experiment.py
├── compare_learning_rates.py
├── run_cross_validation.py
└── run_baseline_comparison.py

results/
├── metrics.csv
├── learning_rate_comparison.csv
├── cross_validation.csv
├── baseline_comparison.csv
└── figures/
    ├── loss_curve.png
    ├── roc_curve.png
    └── confusion_matrix.png
```

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the main experiment:

```bash
python3 experiments/run_experiment.py
```

Compare learning rates:

```bash
python3 experiments/compare_learning_rates.py
```

Run 5-fold cross-validation:

```bash
python3 experiments/run_cross_validation.py
```

Compare against scikit-learn LogisticRegression:

```bash
python3 experiments/run_baseline_comparison.py
```

## Main Experiment Results

From-scratch Logistic Regression on one train/test split:

| Metric | Value |
|---|---:|
| Accuracy | 0.9646 |
| Precision | 0.9494 |
| Recall | 1.0000 |
| Specificity | 0.8947 |
| F1 Score | 0.9740 |
| Test BCE Loss | 0.0925 |
| ROC AUC | 0.9928 |

A small learning-rate comparison was performed, and `learning_rate=0.005` was selected because it achieved the lowest test BCE among the tested values while maintaining the same accuracy and F1 score.

## Baseline Comparison

The from-scratch Logistic Regression implementation was compared against `scikit-learn`'s `LogisticRegression` using the same train/test split and preprocessing pipeline.

| Model | Accuracy | Precision | Recall | F1 Score | ROC AUC |
|---|---:|---:|---:|---:|---:|
| From-scratch Logistic Regression | 0.9646 | 0.9494 | 1.0000 | 0.9740 | 0.9928 |
| scikit-learn LogisticRegression | 0.9646 | 0.9494 | 1.0000 | 0.9740 | 0.9902 |

The from-scratch implementation achieved comparable performance to the scikit-learn baseline on this split.

## Cross-Validation Results

Using 5-fold cross-validation with `learning_rate=0.005`:

| Metric | Mean ± Std |
|---|---:|
| Accuracy | 0.9772 ± 0.0131 |
| Precision | 0.9757 ± 0.0169 |
| Recall | 0.9884 ± 0.0114 |
| Specificity | 0.9561 ± 0.0345 |
| F1 Score | 0.9819 ± 0.0106 |
| BCE Loss | 0.0774 ± 0.0240 |

These results suggest that the model performs consistently across different train/validation splits.

## Visualizations

The experiment saves plots to `results/figures/`.

Current plots:

- Training loss curve
- ROC curve
- Confusion matrix

## Reproducibility

All experiment outputs are saved under the `results/` directory, including metric CSV files and generated figures.

Experiments use fixed random seeds where applicable to make results reproducible.

## Limitations

This project currently focuses on binary classification with Logistic Regression. It is intended as an educational project that emphasizes understanding the algorithm, optimization process, preprocessing, and evaluation workflow.

Potential extensions include adding regularization, additional from-scratch classifiers, unit tests, and evaluation on additional datasets.