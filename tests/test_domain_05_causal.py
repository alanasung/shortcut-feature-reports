from introspect.introspect.activations import synthetic_activations
from introspect.introspect.causal import ablate_direction, paired_activation_patch
from introspect.introspect.data import build_cued_bias_dataset
from introspect.introspect.probes import fit_feature_probes

def test_patch_and_ablate():
    ds = build_cued_bias_dataset(n_items=70, seed=0)
    acts = synthetic_activations(ds["items"], layers=[2], dim=24, seed=0)
    probes = fit_feature_probes(acts, features=["hint_reliance"], layer=2, holdout_feature="planted_topic", seed=0, min_auroc=0.0)
    probe = next(iter(probes["probes"].values()))
    c = paired_activation_patch(acts, probe, layer=2, seed=0)
    assert "causal_sensitivity" in c and "shuffled_null" in c
    a = ablate_direction(acts, probe, layer=2)
    assert a["mean_abs_change"] >= 0
