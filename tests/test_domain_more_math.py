import numpy as np
from introspect.introspect.verbalize import score_verbalization
from introspect.introspect.probes import fit_probe, probe_predict

def test_score_empty():
    s = score_verbalization([])
    assert s["n"] == 0.0

def test_fit_probe_degenerate():
    X = np.ones((5, 3))
    y = np.zeros(5, dtype=int)
    clf = fit_probe(X, y, seed=0)
    p = probe_predict({"coef": clf.coef_.tolist(), "intercept": float(np.asarray(clf.intercept_).reshape(-1)[0])}, X)
    assert len(p) == 5

def test_score_perfect():
    reports = [{"behavioral_gt": 1, "report_active": 1, "confidence": 0.9} for _ in range(10)]
    assert score_verbalization(reports)["accuracy_behavioral"] == 1.0

def test_score_wrong():
    reports = [{"behavioral_gt": 1, "report_active": 0, "confidence": 0.9} for _ in range(10)]
    assert score_verbalization(reports)["accuracy_behavioral"] == 0.0
