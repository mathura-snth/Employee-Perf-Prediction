import pandas as pd
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support,
)

LABELS = [1, 2, 3, 4, 5]


def evaluate(y_true, y_pred, labels=LABELS):
    summary = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0),
    }
    p, r, f, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    per_class = pd.DataFrame(
        {"precision": p, "recall": r, "f1": f, "support": support}, index=labels
    )
    return summary, per_class


def confusion_rows(y_true, y_pred, labels=LABELS):
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")
    return pd.DataFrame(cm, index=labels, columns=labels)
