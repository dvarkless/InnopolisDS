import matplotlib.pyplot as plt
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             precision_score, recall_score, f1_score)


def show_metrics_matrix(preds, answer):
    labels = list("ABCDEFGHIJKLMNOPQRS")
    ConfusionMatrixDisplay.from_predictions(
        answer,
        preds,
        labels,
        normalize="true",
    )
    plt.show()


def return_precision(preds, answer):
    return precision_score(answer, preds, average='macro', zero_division=1)


def return_recall(preds, answer):
    return recall_score(answer, preds, average='macro', zero_division=1)


def return_f1(preds, answer):
    return f1_score(answer, preds, average='macro', zero_division=1)


def return_accuracy(preds, answer):
    return accuracy_score(answer, preds)

def predictions_mean(preds, answer):
    return preds.mean()

def predictions_std(preds, answer):
    return preds.std()

def print_accuracy(preds, answer):
    print(accuracy_score(answer, preds))
