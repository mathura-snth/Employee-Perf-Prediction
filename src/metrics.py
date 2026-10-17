import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support,
)

from src.config import SEED

LABELS = [1, 2, 3, 4, 5]


def evaluate(y_true, y_pred, labels=LABELS):
    """Return (summary, per_class): global scores and a precision/recall/F1 table."""
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
    """Confusion matrix normalized by row: each row sums to 1 (= per-class recall on the diagonal)."""
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")
    return pd.DataFrame(cm, index=labels, columns=labels)


def _scores(y_idx, p_idx, labels):
    """Accuracy, macro F1 and per-class recall from label positions (fast, via a confusion matrix)."""
    K = len(labels)
    cm = np.bincount(y_idx * K + p_idx, minlength=K * K).reshape(K, K)
    tp, support, n_pred = np.diag(cm), cm.sum(axis=1), cm.sum(axis=0)
    f1 = np.divide(2 * tp, support + n_pred, out=np.zeros(K), where=(support + n_pred) > 0)
    recall = np.divide(tp, support, out=np.zeros(K), where=support > 0)
    return {"accuracy": tp.sum() / cm.sum(), "macro_f1": f1.mean(),
            **{f"recall_{c}": recall[i] for i, c in enumerate(labels)}}


def _positions(y_true, y_pred, labels):
    return np.searchsorted(labels, np.asarray(y_true)), np.searchsorted(labels, np.asarray(y_pred))


def bootstrap_ci(y_true, y_pred, labels=LABELS, n_boot=1000, seed=SEED):
    """Point estimates and 95% bootstrap intervals (resampling the test rows).
    Returns a DataFrame indexed by metric with columns value, low, high. The same
    seed gives the same resamples for every model, so results are comparable.
    """
    y_idx, p_idx = _positions(y_true, y_pred, labels)
    rng = np.random.default_rng(seed)
    n = len(y_idx)
    boots = pd.DataFrame([
        _scores(y_idx[i], p_idx[i], labels)
        for i in (rng.integers(0, n, n) for _ in range(n_boot))
    ])
    out = pd.DataFrame({"value": _scores(y_idx, p_idx, labels)})
    out["low"], out["high"] = boots.quantile(0.025), boots.quantile(0.975)
    return out


def bootstrap_diff(y_true, pred_a, pred_b, labels=LABELS, n_boot=1000, seed=SEED):
    """Paired bootstrap of metric(a) - metric(b) on the same resampled rows."""
    y_idx, pa = _positions(y_true, pred_a, labels)
    _, pb = _positions(y_true, pred_b, labels)
    rng = np.random.default_rng(seed)
    n = len(y_idx)
    diffs = pd.DataFrame([
        {k: v - _scores(y_idx[i], pb[i], labels)[k]
         for k, v in _scores(y_idx[i], pa[i], labels).items()}
        for i in (rng.integers(0, n, n) for _ in range(n_boot))
    ])
    point_a, point_b = _scores(y_idx, pa, labels), _scores(y_idx, pb, labels)
    out = pd.DataFrame({"value": {k: point_a[k] - point_b[k] for k in point_a}})
    out["low"], out["high"] = diffs.quantile(0.025), diffs.quantile(0.975)
    return out
