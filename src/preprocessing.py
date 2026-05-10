import numpy as np

def standardize_fit(X):
    """
    Compute feature means and standard deviations from the training data.
    """
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)

    # Avoid division by zero for constant features
    std = np.where(std == 0, 1, std)

    return mean, std


def standardize_transform(X, mean, std):
    """
    Standardize data using precomputed mean and std.
    """
    return (X - mean) / std


def add_bias_term(X):
    """
    Add a column of ones for the bias term.
    """
    bias = np.ones((X.shape[0], 1))
    return np.hstack([bias, X])