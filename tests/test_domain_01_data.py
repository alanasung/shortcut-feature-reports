import pytest
from introspect.introspect.data import FEATURES, behavioral_label, build_cued_bias_dataset, iter_split

def test_feature_count():
    assert len(FEATURES) == 5

@pytest.mark.parametrize("feat", list(FEATURES))
def test_feature_in_dataset(feat):
    ds = build_cued_bias_dataset(n_items=50, seed=0)
    assert any(r["feature"] == feat for r in ds["items"])

def test_holdout_and_size():
    ds = build_cued_bias_dataset(n_items=50, seed=0, holdout_feature="planted_topic")
    assert ds["holdout_feature"] == "planted_topic"
    assert ds["n"] >= 50

def test_unique_ids():
    ds = build_cued_bias_dataset(n_items=50, seed=1)
    ids = [r["item_id"] for r in ds["items"]]
    assert len(ids) == len(set(ids))

def test_iter_split():
    ds = build_cued_bias_dataset(n_items=50, seed=2)
    assert all(r["split"] == "train" for r in iter_split(ds, "train"))

def test_bad_holdout():
    with pytest.raises(ValueError):
        build_cued_bias_dataset(n_items=50, seed=0, holdout_feature="x")

def test_too_small():
    with pytest.raises(ValueError):
        build_cued_bias_dataset(n_items=3, seed=0)

@pytest.mark.parametrize("feat,expected", [
    ("hint_reliance", 1),
    ("answer_certainty", 1),
    ("format_compliance", 0),
    ("planted_topic", 0),
    ("sycophantic_agreement", 1),
])
def test_behavioral(feat, expected):
    row = {"hint_flips_answer": True, "paraphrase_consistency": 0.9, "format_ok": False, "planted_topic": "none", "sycophancy_flips": True}
    assert behavioral_label(feat, row) == expected
