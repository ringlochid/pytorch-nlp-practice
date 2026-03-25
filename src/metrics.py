from __future__ import annotations

from typing import Iterable

from sklearn.metrics import accuracy_score, f1_score


def classification_metrics(y_true: Iterable[int], y_pred: Iterable[int]) -> dict:
    y_true = list(y_true)
    y_pred = list(y_pred)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }
