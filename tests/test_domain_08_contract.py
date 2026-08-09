from introspect.introspect.data import build_cued_bias_dataset
from introspect.introspect.verbalize import score_verbalization, synthetic_verbalize

def test_dataset_note():
    assert "probe" in build_cued_bias_dataset(n_items=40, seed=0)["note"].lower() or "behavioral" in build_cued_bias_dataset(n_items=40, seed=0)["note"].lower()

def test_reports_carry_behavioral_gt():
    ds = build_cued_bias_dataset(n_items=40, seed=0)
    reps = synthetic_verbalize(ds["items"], seed=0)
    assert all("behavioral_gt" in r for r in reps)
    assert 0 <= score_verbalization(reps)["accuracy_behavioral"] <= 1
