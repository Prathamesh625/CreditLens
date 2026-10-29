"""Metrics suited to imbalanced credit data."""
from __future__ import annotations

import numpy as np
from scipy.stats import ks_2samp
from sklearn.metrics import average_precision_score, roc_auc_score


def ks_statistic(y_true, y_score) -> float:
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    return float(ks_2samp(y_score[y_true == 1], y_score[y_true == 0]).statistic)


def evaluate(y_true, y_score) -> dict:
    return {
        "roc_auc": round(float(roc_auc_score(y_true, y_score)), 4),
        "pr_auc": round(float(average_precision_score(y_true, y_score)), 4),
        "ks": round(ks_statistic(y_true, y_score), 4),
    }
