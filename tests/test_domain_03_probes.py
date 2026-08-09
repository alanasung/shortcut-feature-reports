import numpy as np
from introspect.introspect.activations import synthetic_activations
from introspect.introspect.data import FEATURES, build_cued_bias_dataset
from introspect.introspect.probes import fit_feature_probes, fit_probe, probe_predict, validate_probe_against_behavior

def _b():
    ds = build_cued_bias_dataset(n_items=80, seed=0)
    return ds, synthetic_activations(ds["items"], layers=[3], dim=32, seed=0)

def test_validate_keys():
    _, acts = _b()
    v = validate_probe_against_behavior(acts, feature="hint_reliance", layer=3, seed=0, min_auroc=0.5)
    assert {"auroc", "accepted", "feature"} <= set(v)

def test_fit_bundle():
    ds, acts = _b()
    out = fit_feature_probes(acts, features=list(FEATURES), layer=3, holdout_feature=ds["holdout_feature"], seed=0, min_auroc=0.0)
    assert "probes" in out and "dropped_features" in out

def test_predict_bounds():
    ds, acts = _b()
    out = fit_feature_probes(acts, features=["hint_reliance"], layer=3, holdout_feature="planted_topic", seed=0, min_auroc=0.0)
    if "hint_reliance" in out["probes"]:
        p = probe_predict(out["probes"]["hint_reliance"], np.asarray(acts["activations"]["3"][:4]))
        assert np.all((p >= 0) & (p <= 1))

def test_degenerate_probe():
    X = np.ones((6, 4)); y = np.zeros(6, dtype=int)
    clf = fit_probe(X, y, seed=0)
    assert clf.coef_.shape[1] == 4
