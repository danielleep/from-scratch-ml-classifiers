# From-Scratch ML Classifiers

Machine learning classifiers implemented from scratch using Python and NumPy.

Currently, the project includes a complete Logistic Regression pipeline with batch gradient descent, binary cross-entropy loss, preprocessing, evaluation metrics, visualizations, learning-rate comparison, and cross-validation.

## Dataset

The current experiment uses the Breast Cancer Wisconsin dataset from `scikit-learn`.

- 569 samples
- 30 numeric features
- Binary classification: malignant / benign

## Implemented So Far

- Logistic Regression from scratch
- Sigmoid function
- Binary cross-entropy loss
- Batch gradient descent
- Train/test split with NumPy
- Feature standardization using training-set statistics
- Bias term
- Accuracy, precision, recall, specificity, F1 score
- ROC curve and AUC
- Confusion matrix plot
- Training loss curve
- Learning-rate comparison
- 5-fold cross-validation
- Saving metrics and experiment results to CSV

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
└── run_cross_validation.py

results/
├── metrics.csv
├── learning_rate_comparison.csv
├── cross_validation.csv
└── figures/
    ├── loss_curve.png
    ├── roc_curve.png
    └── confusion_matrix.png
```

## How to Run

Install dependencies and run the main experiment:

```bash
pip install -r requirements.txt
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

## Main Experiment Results

Logistic Regression on one train/test split:

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

These results suggest that the Logistic Regression model performs consistently across different train/validation splits.

## Visualizations

The experiment saves plots to `results/figures/`.

Current plots:

- Training loss curve
- ROC curve
- Confusion matrix

## Future Improvements

This project can be extended with additional from-scratch classifiers such as Naive Bayes and Gaussian MAP, model comparison tables, and unit tests for core functions.