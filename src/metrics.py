import numpy as np


def confusion_matrix_binary(y_true, y_pred, positive_class=1):
    """
    Compute TP, FP, TN, FN for binary classification.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    y_true_binary = (y_true == positive_class).astype(int)
    y_pred_binary = (y_pred == positive_class).astype(int)

    tp = np.sum((y_true_binary == 1) & (y_pred_binary == 1))
    fp = np.sum((y_true_binary == 0) & (y_pred_binary == 1))
    tn = np.sum((y_true_binary == 0) & (y_pred_binary == 0))
    fn = np.sum((y_true_binary == 1) & (y_pred_binary == 0))

    return tp, fp, tn, fn


def accuracy_score(y_true, y_pred):
    """
    Compute accuracy.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    return np.mean(y_true == y_pred)


def precision_score(y_true, y_pred, positive_class=1):
    """
    Compute precision: TP / (TP + FP).
    """
    tp, fp, tn, fn = confusion_matrix_binary(y_true, y_pred, positive_class)

    if tp + fp == 0:
        return 0.0

    return tp / (tp + fp)


def recall_score(y_true, y_pred, positive_class=1):
    """
    Compute recall / sensitivity: TP / (TP + FN).
    """
    tp, fp, tn, fn = confusion_matrix_binary(y_true, y_pred, positive_class)

    if tp + fn == 0:
        return 0.0

    return tp / (tp + fn)


def specificity_score(y_true, y_pred, positive_class=1):
    """
    Compute specificity: TN / (TN + FP).
    """
    tp, fp, tn, fn = confusion_matrix_binary(y_true, y_pred, positive_class)

    if tn + fp == 0:
        return 0.0

    return tn / (tn + fp)


def f1_score(y_true, y_pred, positive_class=1):
    """
    Compute F1 score.
    """
    precision = precision_score(y_true, y_pred, positive_class)
    recall = recall_score(y_true, y_pred, positive_class)

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)


def classification_report_binary(y_true, y_pred, positive_class=1):
    """
    Return all main binary classification metrics in a dictionary.
    """
    tp, fp, tn, fn = confusion_matrix_binary(y_true, y_pred, positive_class)

    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, positive_class),
        "recall": recall_score(y_true, y_pred, positive_class),
        "specificity": specificity_score(y_true, y_pred, positive_class),
        "f1": f1_score(y_true, y_pred, positive_class),
    }