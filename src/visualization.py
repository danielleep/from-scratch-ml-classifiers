import matplotlib.pyplot as plt


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