"""Linear probes validated against behavioral ground truth before use."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold


def _xy(
    bundle: dict[str, Any],
    *,
    feature: str | None,
    layer: int,
    split: str | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    xs: list[list[float]] = []
    ys: list[int] = []
    ids: list[str] = []
    acts = bundle["activations"][str(layer)]
    for meta, vec in zip(bundle["meta"], acts):
        if feature is not None and meta["feature"] != feature:
            continue
        if split is not None and meta["split"] != split:
            continue
        xs.append(vec)
        ys.append(int(meta["behavioral_gt"]))
        ids.append(meta["item_id"])
    return np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.int64), ids


def fit_probe(X: np.ndarray, y: np.ndarray, *, seed: int = 0) -> LogisticRegression:
    clf = LogisticRegression(max_iter=500, random_state=seed)
    if len(X) == 0:
        raise ValueError("cannot fit probe on empty matrix")
    if len(np.unique(y)) < 2:
        clf.classes_ = np.array([0, 1])
        clf.coef_ = np.zeros((1, X.shape[1]))
        clf.intercept_ = np.array([0.0])
        clf.n_features_in_ = X.shape[1]
        return clf
    clf.fit(X, y)
    return clf


def validate_probe_against_behavior(
    bundle: dict[str, Any],
    *,
    feature: str,
    layer: int,
    seed: int = 0,
    min_auroc: float = 0.6,
) -> dict[str, Any]:
    # Train-only validation to avoid selection leakage into the locked test set.
    X, y, _ = _xy(bundle, feature=feature, layer=layer, split="train")
    if len(y) < 8:
        X, y, _ = _xy(bundle, feature=feature, layer=layer)
    if len(y) < 8 or len(np.unique(y)) < 2:
        return {
            "feature": feature,
            "layer": layer,
            "accepted": False,
            "reason": "insufficient class support",
            "auroc": float("nan"),
            "accuracy": float("nan"),
        }
    n_splits = max(2, min(5, len(y) // 4))
    while n_splits > 1:
        counts = np.bincount(y)
        if counts.min() >= n_splits:
            break
        n_splits -= 1
    if n_splits < 2:
        clf = fit_probe(X, y, seed=seed)
        proba = probe_predict(
            {"coef": clf.coef_.tolist(), "intercept": float(clf.intercept_[0])}, X
        )
        try:
            auroc = float(roc_auc_score(y, proba))
        except ValueError:
            auroc = float("nan")
        acc = float(accuracy_score(y, (proba >= 0.5).astype(int)))
        accepted = bool(auroc >= min_auroc) if auroc == auroc else False
        return {
            "feature": feature,
            "layer": layer,
            "accepted": accepted,
            "auroc": auroc,
            "accuracy": acc,
            "reason": "ok" if accepted else "auroc below threshold",
        }

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores: list[float] = []
    accs: list[float] = []
    for tr, te in cv.split(X, y):
        clf = fit_probe(X[tr], y[tr], seed=seed)
        probe = {"coef": clf.coef_.tolist(), "intercept": float(clf.intercept_[0])}
        proba = probe_predict(probe, X[te])
        try:
            scores.append(float(roc_auc_score(y[te], proba)))
        except ValueError:
            scores.append(float("nan"))
        accs.append(float(accuracy_score(y[te], (proba >= 0.5).astype(int))))
    auroc = float(np.nanmean(scores))
    acc = float(np.nanmean(accs))
    accepted = bool(auroc >= min_auroc)
    return {
        "feature": feature,
        "layer": layer,
        "accepted": accepted,
        "auroc": auroc,
        "accuracy": acc,
        "reason": "ok" if accepted else f"auroc {auroc:.3f} < {min_auroc}",
    }


def fit_feature_probes(
    bundle: dict[str, Any],
    *,
    features: list[str],
    layer: int,
    holdout_feature: str,
    seed: int = 0,
    min_auroc: float = 0.6,
) -> dict[str, Any]:
    validations = []
    probes: dict[str, Any] = {}
    dropped: list[str] = []
    for feature in features:
        val = validate_probe_against_behavior(
            bundle, feature=feature, layer=layer, seed=seed, min_auroc=min_auroc
        )
        validations.append(val)
        if not val["accepted"]:
            dropped.append(feature)
            continue
        # Held-out FEATURE: fit a diagnostic probe on that feature's train split only
        # (never the locked test split). Non-holdout probes train on train split.
        X, y, _ = _xy(bundle, feature=feature, layer=layer, split="train")
        if len(y) < 4:
            dropped.append(feature)
            continue
        clf = fit_probe(X, y, seed=seed)
        probes[feature] = {
            "coef": np.asarray(clf.coef_, dtype=float).tolist(),
            "intercept": float(np.asarray(clf.intercept_).reshape(-1)[0]),
            "layer": layer,
            "trained_on_holdout_feature": feature == holdout_feature,
            "fit_split": "train",
        }
    return {
        "layer": layer,
        "probes": probes,
        "validations": validations,
        "dropped_features": dropped,
        "holdout_feature": holdout_feature,
        "note": (
            "probe labels are training diagnostics; behavioral_gt is evaluation GT; "
            "probes fit on train split only"
        ),
    }


def probe_predict(probe: dict[str, Any], X: np.ndarray) -> np.ndarray:
    coef = np.asarray(probe["coef"], dtype=np.float64)
    logits = X @ coef.T + float(probe["intercept"])
    if logits.ndim > 1:
        logits = logits[:, 0]
    return 1.0 / (1.0 + np.exp(-logits))
