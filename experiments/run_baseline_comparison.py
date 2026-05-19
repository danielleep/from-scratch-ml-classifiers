import sys
import csv
from pathlib import Path

# Make sure Python can import files from the project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from sklearn.linear_model import LogisticRegression

from src.data import load_breast_cancer_dataset, train_test_split_numpy
from src.preprocessing import standardize_fit, standardize_transform, add_bias_term
from src.logistic_regression import LogisticRegressionGD
from src.metrics import classification_report_binary, roc_curve_points, auc_score


def evaluate_model(y_true, y_pred, y_prob, positive_class=1):
    """
    Compute the main evaluation metrics for a binary classifier.
    """
    report = classification_report_binary(
        y_true,
        y_pred,
        positive_class=positive_class
    )

    fpr, tpr = roc_curve_points(
        y_true,
        y_prob,
        positive_class=positive_class
    )

    roc_auc = auc_score(fpr, tpr)

    return {
        "accuracy": report["accuracy"],
        "precision": report["precision"],
        "recall": report["recall"],
        "f1": report["f1"],
        "roc_auc": roc_auc,
    }


def main():
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(exist_ok=True)

    # Load and split data
    X, y, feature_names, target_names = load_breast_cancer_dataset()

    X_train, X_test, y_train, y_test = train_test_split_numpy(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # Fit preprocessing only on training data
    mean, std = standardize_fit(X_train)

    X_train_scaled = standardize_transform(X_train, mean, std)
    X_test_scaled = standardize_transform(X_test, mean, std)

    # From-scratch model uses an explicit bias column
    X_train_bias = add_bias_term(X_train_scaled)
    X_test_bias = add_bias_term(X_test_scaled)

    # Train from-scratch Logistic Regression
    scratch_model = LogisticRegressionGD(
        learning_rate=0.005,
        max_iter=10000,
        eps=1e-6,
        random_state=42
    )

    scratch_model.fit(X_train_bias, y_train)

    scratch_pred = scratch_model.predict(X_test_bias)
    scratch_prob = scratch_model.predict_proba(X_test_bias)

    scratch_metrics = evaluate_model(
        y_test,
        scratch_pred,
        scratch_prob,
        positive_class=1
    )

    # Train scikit-learn Logistic Regression baseline
    sklearn_model = LogisticRegression(
        max_iter=10000,
        random_state=42
    )

    sklearn_model.fit(X_train_scaled, y_train)

    sklearn_pred = sklearn_model.predict(X_test_scaled)
    sklearn_prob = sklearn_model.predict_proba(X_test_scaled)[:, 1]

    sklearn_metrics = evaluate_model(
        y_test,
        sklearn_pred,
        sklearn_prob,
        positive_class=1
    )

    # Save comparison results
    output_path = results_dir / "baseline_comparison.csv"

    rows = [
        {
            "model": "From-scratch Logistic Regression",
            **scratch_metrics
        },
        {
            "model": "scikit-learn LogisticRegression",
            **sklearn_metrics
        }
    ]

    with open(output_path, mode="w", newline="") as file:
        fieldnames = [
            "model",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc"
        ]

        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("Baseline comparison")
    print("-" * 60)

    for row in rows:
        print(f"\n{row['model']}")
        print(f"Accuracy: {row['accuracy']:.4f}")
        print(f"Precision: {row['precision']:.4f}")
        print(f"Recall: {row['recall']:.4f}")
        print(f"F1 Score: {row['f1']:.4f}")
        print(f"ROC AUC: {row['roc_auc']:.4f}")

    print(f"\nSaved baseline comparison to: {output_path}")


if __name__ == "__main__":
    main()