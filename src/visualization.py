import matplotlib.pyplot as plt
import numpy as np

def plot_loss_curve(loss_history, save_path):
    """
    Plot and save the training loss curve.
    """
    plt.figure(figsize=(8, 5))

    plt.plot(loss_history)

    plt.xlabel("Iteration")
    plt.ylabel("BCE Loss")
    plt.title("Logistic Regression Training Loss")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_confusion_matrix(tp, fp, tn, fn, save_path):
    """
    Plot and save a binary confusion matrix.
    """
    matrix = np.array([
        [tn, fp],
        [fn, tp]
    ])

    plt.figure(figsize=(5, 4))
    plt.imshow(matrix)

    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    plt.xticks([0, 1], ["Negative", "Positive"])
    plt.yticks([0, 1], ["Negative", "Positive"])

    for i in range(2):
        for j in range(2):
            plt.text(j, i, matrix[i, j], ha="center", va="center")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_roc_curve(fpr, tpr, auc, save_path):
    """
    Plot and save the ROC curve.
    """
    plt.figure(figsize=(6, 5))

    plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()