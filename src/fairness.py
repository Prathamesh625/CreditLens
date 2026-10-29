"""Simple fairness audit: compare decisions and error rates across groups."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import DECISION_THRESHOLD, TARGET


def _rates(y, flagged) -> dict:
    y = np.asarray(y)
    flagged = np.asarray(flagged)
    neg, pos = (y == 0), (y == 1)
    return {
        "n": int(len(y)),
        "approval_rate": round(float((~flagged).mean()), 4),
        # good customer wrongly rejected
        "false_positive_rate": round(float(flagged[neg].mean()), 4) if neg.any() else None,
        # defaulter wrongly approved
        "false_negative_rate": round(float((~flagged[pos]).mean()), 4) if pos.any() else None,
    }


def audit(df_test: pd.DataFrame, scores, threshold: float = DECISION_THRESHOLD) -> dict:
    df = df_test.copy()
    df["flagged"] = np.asarray(scores) >= threshold
    df["age_band"] = pd.cut(df["age"], [0, 29, 49, 200], labels=["<30", "30-49", "50+"])
    out: dict = {"threshold": threshold, "groups": {}}
    for col in ("gender", "age_band"):
        if col not in df.columns:
            continue
        out["groups"][col] = {
            str(g): _rates(sub[TARGET], sub["flagged"])
            for g, sub in df.groupby(col, observed=True)
        }
        rates = [v["approval_rate"] for v in out["groups"][col].values()]
        out["groups"][col]["_approval_rate_gap"] = round(float(max(rates) - min(rates)), 4)
    return out
