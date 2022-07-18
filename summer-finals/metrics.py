import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt


def show_metrics_matrix(preds, ans):
    labels = list("ABCDEFGHIJKLMNOPQRS")
    ConfusionMatrixDisplay.from_predictions(
        preds,
        ans,
        labels,
        normalize="true",
    )
    plt.show()


def return_accuracy(preds, ans):
    return np.count_nonzero(preds == ans) / len(ans)


def print_accuracy(preds, ans):
    print(np.count_nonzero(preds == ans) / len(ans))
