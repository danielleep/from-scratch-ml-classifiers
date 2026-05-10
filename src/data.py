import numpy as np
from sklearn.datasets import load_breast_cancer

def load_breast_cancer_dataset():
    """
    Load the Breast Cancer Wisconsin dataset.

    Returns
    -------
    X : ndarray, shape = [n_samples, n_features]
        Feature matrix.

    y : ndarray, shape = [n_samples]
        Binary class labels.

    feature_names : ndarray
        Names of the input features.

    target_names : ndarray
        Names of the target classes.
    """
    data = load_breast_cancer()

    X = data.data
    y = data.target
    feature_names = data.feature_names
    target_names = data.target_names

    return X, y, feature_names, target_names


def train_test_split_numpy(X, y, test_size=0.2, random_state=42):
    """
    Split X and y into train and test sets using NumPy.
    """
    if not 0 < test_size < 1:
        raise ValueError("test_size must be a float between 0 and 1.")

    n_samples = X.shape[0]

    rng = np.random.default_rng(random_state)
    indices = rng.permutation(n_samples)

    test_count = int(n_samples * test_size)

    test_indices = indices[:test_count]
    train_indices = indices[test_count:]

    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]

    return X_train, X_test, y_train, y_test