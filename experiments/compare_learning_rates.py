import sys
import csv
from pathlib import Path

# Make sure Python can import files from the project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.data import load_breast_cancer_dataset, train_test_split_numpy
from src.preprocessing import standardize_fit, standardize_transform, add_bias_term
from src.logistic_regression import LogisticRegressionGD
from src.metrics import classification_report_binary


def main():
    # Create results folder if needed
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(exist_ok=True)

    # Load and split the dataset
    X, y, feature_names, target_names = load_breast_cancer_dataset()

    X_train, X_test, y_train, y_test = train_test_split_numpy(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # Standardize using training data only
    mean, std = standardize_fit(X_train)

    X_train_scaled = standardize_transform(X_train, mean, std)
    X_test_scaled = standardize_transform(X_test, mean, std)

    # Add bias term for logistic regression
    X_train_bias = add_bias_term(X_train_scaled)
    X_test_bias = add_bias_term(X_test_scaled)

    # Learning rates to compare
    learning_rates = [0.001, 0.005, 0.01, 0.05, 0.1]

    results = []

    for lr in learning_rates:
        print(f"\nTraining model with learning_rate={lr}")

        model = LogisticRegressionGD(
            learning_rate=lr,
            max_iter=10000,
            eps=1e-6,
            random_state=42
        )

        model.fit(X_train_bias, y_train)

        y_pred = model.predict(X_test_bias)

        report = classification_report_binary(
            y_test,
            y_pred,
            positive_class=1
        )

        test_bce = model.bce_loss(X_test_bias, y_test)
        final_train_loss = model.loss_history_[-1]

        result = {
            "learning_rate": lr,
            "accuracy": report["accuracy"],
            "precision": report["precision"],
            "recall": report["recall"],
            "specificity": report["specificity"],
            "f1": report["f1"],
            "test_bce": test_bce,
            "final_train_loss": final_train_loss,
            "n_iter": model.n_iter_
        }

        results.append(result)

        print(f"Accuracy: {report['accuracy']:.4f}")
        print(f"F1: {report['f1']:.4f}")
        print(f"Test BCE: {test_bce:.4f}")
        print(f"Iterations: {model.n_iter_}")

    # Save comparison results
    output_path = results_dir / "learning_rate_comparison.csv"

    with open(output_path, mode="w", newline="") as file:
        fieldnames = [
            "learning_rate",
            "accuracy",
            "precision",
            "recall",
            "specificity",
            "f1",
            "test_bce",
            "final_train_loss",
            "n_iter"
        ]

        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nLearning rate comparison saved to: {output_path}")

    # Select best learning rate by F1 score, then by lower test BCE
    best_result = sorted(
        results,
        key=lambda row: (-row["f1"], row["test_bce"])
    )[0]

    print("\nBest learning rate based on F1 and test BCE:")
    print(f"learning_rate: {best_result['learning_rate']}")
    print(f"F1: {best_result['f1']:.4f}")
    print(f"Test BCE: {best_result['test_bce']:.4f}")
    print(f"Accuracy: {best_result['accuracy']:.4f}")


if __name__ == "__main__":
    main()