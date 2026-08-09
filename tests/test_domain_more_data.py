import pytest
from introspect.introspect.data import FEATURES, behavioral_label, build_cued_bias_dataset, iter_split

@pytest.mark.parametrize("feat", FEATURES)
def test_each_feature_present(feat):
    ds = build_cued_bias_dataset(n_items=50, seed=0)
    assert any(r["feature"] == feat for r in ds["items"])

@pytest.mark.parametrize("feat,row,expected", [
    ("hint_reliance", {"hint_flips_answer": False, "paraphrase_consistency": 0.5, "format_ok": True, "planted_topic": "astronomy", "sycophancy_flips": False}, 0),
    ("planted_topic", {"hint_flips_answer": False, "paraphrase_consistency": 0.5, "format_ok": True, "planted_topic": "astronomy", "sycophancy_flips": False}, 1),
    ("answer_certainty", {"hint_flips_answer": False, "paraphrase_consistency": 0.74, "format_ok": True, "planted_topic": "none", "sycophancy_flips": False}, 0),
])
def test_behavioral_cases(feat, row, expected):
    assert behavioral_label(feat, row) == expected

def test_iter_split_filters():
    ds = build_cued_bias_dataset(n_items=50, seed=3)
    train = list(iter_split(ds, "train"))
    assert all(r["split"] == "train" for r in train)

def test_holdout_invalid():
    with pytest.raises(ValueError):
        build_cued_bias_dataset(n_items=50, seed=0, holdout_feature="nope")

def test_item_ids_unique():
    ds = build_cued_bias_dataset(n_items=50, seed=0)
    ids = [r["item_id"] for r in ds["items"]]
    assert len(ids) == len(set(ids))
