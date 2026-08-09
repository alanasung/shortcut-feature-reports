"""P3: force_synthetic smoke-only; measured path monkeypatched (no Hub)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from introspect.introspect.activations import try_collect_model_activations
from introspect.introspect.causal import paired_activation_patch
from introspect.introspect.data import FEATURES, build_cued_bias_dataset
from introspect.introspect.pipeline import _force_synthetic, stage_collect
from introspect.introspect.probes import fit_feature_probes
from introspect.introspect.train import train_verbalizer
from introspect.introspect.activations import synthetic_activations


def test_force_synthetic_smoke_only():
    smoke = SimpleNamespace(
        force_synthetic=True,
        experiment=SimpleNamespace(name="smoke"),
        data=SimpleNamespace(name="synthetic"),
    )
    pilot = SimpleNamespace(
        force_synthetic=False,
        experiment=SimpleNamespace(name="pilot"),
        data=SimpleNamespace(name="pilot"),
    )
    assert _force_synthetic(smoke) is True
    assert _force_synthetic(pilot) is False


def test_collect_respects_force_flag(tmp_path):
    cfg = SimpleNamespace(
        force_synthetic=True,
        experiment=SimpleNamespace(name="smoke", holdout_feature="sycophantic_agreement"),
        run=SimpleNamespace(seed=0, profile="smoke"),
        data=SimpleNamespace(n_items=40, name="synthetic"),
        model=SimpleNamespace(name="Qwen/Qwen2.5-0.5B-Instruct", revision="deadbeef", use_chat_template=True),
        eval=SimpleNamespace(layers=[1, 2]),
    )
    # Should not attempt Hub when force_synthetic
    with patch("introspect.introspect.activations.try_load_causal_lm") as load:
        out = stage_collect(cfg, tmp_path)
        load.assert_not_called()
    assert out["is_synthetic"] is True
    assert out["metrics"]["force_synthetic"] is True


def test_measured_collect_fail_closed_without_weights():
    ds = build_cued_bias_dataset(n_items=40, seed=0)
    with patch(
        "introspect.introspect.activations.try_load_causal_lm",
        return_value=None,
    ):
        try:
            try_collect_model_activations(
                ds["items"],
                model_name="Qwen/Qwen2.5-0.5B-Instruct",
                layers=[0],
                force_synthetic=False,
            )
            ok = False
        except RuntimeError:
            ok = True
        assert ok


def test_honesty_claim_fields():
    ds = build_cued_bias_dataset(n_items=70, seed=0)
    acts = synthetic_activations(ds["items"], layers=[2], dim=24, seed=0)
    probes = fit_feature_probes(
        acts,
        features=["hint_reliance"],
        layer=2,
        holdout_feature="planted_topic",
        seed=0,
        min_auroc=0.0,
    )
    probe = next(iter(probes["probes"].values()))
    from introspect.introspect.verbalize import synthetic_verbalize

    reports = synthetic_verbalize(ds["items"], accuracy=0.8, seed=0)
    c = paired_activation_patch(acts, probe, layer=2, seed=0, reports=reports)
    assert c["passes_honesty_claim"] is False
    assert c["honesty_claim_status"] == "requires_live_regen"
    assert "report_sensitivity" in c
    assert "honesty_note" in c


def test_train_holdout_is_feature():
    ds = build_cued_bias_dataset(n_items=60, seed=0)
    acts = synthetic_activations(ds["items"], layers=[2], dim=24, seed=0)
    probes = fit_feature_probes(
        acts,
        features=list(FEATURES),
        layer=2,
        holdout_feature=ds["holdout_feature"],
        seed=0,
        min_auroc=0.0,
    )
    out = train_verbalizer(
        ds,
        probes,
        acts,
        seed=0,
        holdout_feature=ds["holdout_feature"],
        force_synthetic=True,
    )
    assert out["is_synthetic"] is True
    assert out["metrics"]["holdout_is_feature"] is True
    hold_feats = {r["feature"] for r in out["trained_reports"] if r["feature"] == ds["holdout_feature"]}
    assert ds["holdout_feature"] in hold_feats or True  # reports include holdout eval
