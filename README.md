# From-Scratch ML Classifiers

Machine learning classifiers implemented from scratch using Python and NumPy.

Currently, the project includes a complete Logistic Regression pipeline with batch gradient descent, binary cross-entropy loss, preprocessing, and evaluation metrics.

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

## Project Structure

```text
src/
├── data.py
├── preprocessing.py
├── logistic_regression.py
└── metrics.py

experiments/
└── run_experiment.py
```

## How to Run

```bash
pip install -r requirements.txt
python3 experiments/run_experiment.py
```

## Initial Results

| Metric | Value |
|---|---:|
| Accuracy | 0.9646 |
| Precision | 0.9494 |
| Recall | 1.0000 |
| Specificity | 0.8947 |
| F1 Score | 0.9740 |
| Test BCE Loss | 0.0959 |

These are preliminary results using a fixed learning rate and one train/test split.
