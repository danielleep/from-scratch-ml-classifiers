import sys
from pathlib import Path

# Make sure Python can import files from the project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.data import load_breast_cancer_dataset, train_test_split_numpy
from src.preprocessing import standardize_fit, standardize_transform, add_bias_term
from src.logistic_regression import LogisticRegressionGD
from src.metrics import classification_report_binary


def main():
    # Load the dataset
    X, y, feature_names, target_names = load_breast_cancer_dataset()

    print("Dataset loaded")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"Target names: {target_names}")

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split_numpy(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    print("\nTrain/test split")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")

    # Compute scaling parameters only from the training data
    mean, std = standardize_fit(X_train)

    # Apply the same scaling to train and test
    X_train_scaled = standardize_transform(X_train, mean, std)
    X_test_scaled = standardize_transform(X_test, mean, std)

    # Add bias column for the intercept term
    X_train_bias = add_bias_term(X_train_scaled)
    X_test_bias = add_bias_term(X_test_scaled)

    print("\nAfter preprocessing")
    print(f"X_train_bias shape: {X_train_bias.shape}")
    print(f"X_test_bias shape: {X_test_bias.shape}")

    # Create and train the logistic regression model
    model = LogisticRegressionGD(
        learning_rate=0.01,
        max_iter=10000,
        eps=1e-6,
        random_state=42
    )

    model.fit(X_train_bias, y_train)

    print("\nModel trained")
    print(f"Number of iterations: {model.n_iter_}")
    print(f"Final training BCE loss: {model.loss_history_[-1]:.4f}")

    # Predict labels for the test set
    y_pred = model.predict(X_test_bias)

    # Evaluate predictions
    report = classification_report_binary(
        y_test,
        y_pred,
        positive_class=1
    )

    test_bce = model.bce_loss(X_test_bias, y_test)

    print("\nTest results")
    print(f"TP: {report['tp']}")
    print(f"FP: {report['fp']}")
    print(f"TN: {report['tn']}")
    print(f"FN: {report['fn']}")
    print(f"Accuracy: {report['accuracy']:.4f}")
    print(f"Precision: {report['precision']:.4f}")
    print(f"Recall: {report['recall']:.4f}")
    print(f"Specificity: {report['specificity']:.4f}")
    print(f"F1 score: {report['f1']:.4f}")
    print(f"Test BCE loss: {test_bce:.4f}")


if __name__ == "__main__":
    main()