from introspect.introspect.activations import synthetic_activations, try_collect_model_activations
from introspect.introspect.data import build_cued_bias_dataset

def test_synth_dims():
    ds = build_cued_bias_dataset(n_items=40, seed=0)
    b = synthetic_activations(ds["items"], layers=[2, 4], dim=32, seed=0)
    assert len(b["activations"]["2"]) == len(ds["items"])
    assert len(b["activations"]["2"][0]) == 32

def test_force_synth():
    ds = build_cued_bias_dataset(n_items=40, seed=0)
    b = try_collect_model_activations(ds["items"], model_name="no/model", layers=[0], force_synthetic=True)
    assert b["mode"] == "synthetic"

def test_meta_ids():
    ds = build_cued_bias_dataset(n_items=40, seed=1)
    b = synthetic_activations(ds["items"], layers=[1], dim=8, seed=1)
    assert [m["item_id"] for m in b["meta"]] == [r["item_id"] for r in ds["items"]]

def test_missing_weights_fail_closed():
    ds = build_cued_bias_dataset(n_items=40, seed=0)
    try:
        try_collect_model_activations(
            ds["items"],
            model_name="definitely/missing-weights-xyz",
            layers=[0],
            force_synthetic=False,
        )
        raised = False
    except RuntimeError as exc:
        raised = True
        assert "refused synthetic" in str(exc).lower() or "could not load" in str(exc).lower()
    assert raised
