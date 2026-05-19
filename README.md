# From-Scratch ML Classifiers

Machine learning classifiers implemented from scratch using Python and NumPy.

Currently, the project includes a complete Logistic Regression pipeline with batch gradient descent, binary cross-entropy loss, preprocessing, evaluation metrics, and visualizations.

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
- Feature standardization
- Bias term
- Accuracy, precision, recall, specificity, F1 score
- ROC curve and AUC
- Training loss curve
- Learning-rate comparison
- Saving metrics to CSV

## Project Structure

```text
src/
├── data.py
├── preprocessing.py
├── logistic_regression.py
├── metrics.py
└── visualization.py

experiments/
├── run_experiment.py
└── compare_learning_rates.py

results/
├── metrics.csv
├── learning_rate_comparison.csv
└── figures/
    ├── loss_curve.png
    ├── roc_curve.png
    └── confusion_matrix.png 
```

## How to Run

```bash
pip install -r requirements.txt
python3 experiments/run_experiment.py
```

To compare learning rates:

```bash
python3 experiments/compare_learning_rates.py
```

## Initial Results

Logistic Regression on the Breast Cancer Wisconsin dataset:

| Metric | Value |
|---|---:|
| Accuracy | 0.9646 |
| Precision | 0.9494 |
| Recall | 1.0000 |
| Specificity | 0.8947 |
| F1 Score | 0.9740 |
| Test BCE Loss | 0.0925 |
| ROC AUC | 0.9928 |

These are preliminary results using a single train/test split.  
A small learning-rate comparison was performed, and `learning_rate=0.005` was selected because it achieved the lowest test BCE among the tested values while maintaining the same accuracy and F1 score.

## Visualizations

The experiment saves visualizations to `results/figures/`.

Current plots:

- Training loss curve
- ROC curve

## Future Improvements

This project can be extended with k-fold cross-validation, additional from-scratch classifiers such as Naive Bayes and Gaussian MAP, model comparison tables, and unit tests for core functions.