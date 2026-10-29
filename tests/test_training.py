import json
import os

from src.metrics import ks_statistic


def test_artifacts_and_metrics(model_dir):
    for name in ("model.joblib", "metrics.json", "fairness.json"):
        assert os.path.exists(os.path.join(model_dir, name))
    m = json.load(open(os.path.join(model_dir, "metrics.json")))
    assert m["test"]["lightgbm"]["roc_auc"] > 0.6
    assert 0 < m["default_rate"] < 0.5


def test_ks_perfect_separation():
    assert ks_statistic([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0


def test_fairness_report_has_groups(model_dir):
    f = json.load(open(os.path.join(model_dir, "fairness.json")))
    assert "gender" in f["groups"] and "age_band" in f["groups"]
