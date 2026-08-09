from introspect.introspect.activations import synthetic_activations
from introspect.introspect.data import FEATURES, build_cued_bias_dataset
from introspect.introspect.probes import fit_feature_probes
from introspect.introspect.train import train_verbalizer
from introspect.introspect.verbalize import build_report_prompt, parse_report, score_verbalization, synthetic_verbalize

def test_parse():
    assert parse_report("ACTIVE=no; CONF=0.25")["active"] is False

def test_baselines():
    ds = build_cued_bias_dataset(n_items=50, seed=0)
    a = score_verbalization(synthetic_verbalize(ds["items"], accuracy=0.9, seed=0))
    b = score_verbalization(synthetic_verbalize(ds["items"], seed=0, baseline="input_only"))
    assert a["accuracy_behavioral"] >= b["accuracy_behavioral"]

def test_prompt():
    ds = build_cued_bias_dataset(n_items=40, seed=0)
    assert "ACTIVE=" in build_report_prompt(ds["items"][0])

def test_train_gap():
    ds = build_cued_bias_dataset(n_items=60, seed=0)
    acts = synthetic_activations(ds["items"], layers=[2], dim=24, seed=0)
    probes = fit_feature_probes(acts, features=list(FEATURES), layer=2, holdout_feature=ds["holdout_feature"], seed=0, min_auroc=0.0)
    out = train_verbalizer(ds, probes, acts, seed=0, holdout_feature=ds["holdout_feature"])
    assert out["metrics"]["trained_seen"]["accuracy_behavioral"] >= out["metrics"]["trained_holdout"]["accuracy_behavioral"] - 1e-9

def test_score_empty():
    assert score_verbalization([])["n"] == 0.0
