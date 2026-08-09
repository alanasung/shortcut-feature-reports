"""P7: multi-feature live GT, powered honesty, patch-position ablation."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from introspect.introspect.activations import synthetic_activations
from introspect.introspect.causal import paired_activation_patch
from introspect.introspect.data import (
    apply_live_behavioral_gt,
    build_cued_bias_dataset,
    is_live_gt_source,
    parse_format_ok,
)
from introspect.introspect.probes import fit_feature_probes
from introspect.introspect.verbalize import synthetic_verbalize


def test_parse_format_ok():
    assert parse_format_ok("ANSWER: B")
    assert parse_format_ok("  ANSWER: a\n")
    assert not parse_format_ok("The answer is B")
    assert not parse_format_ok("")


def test_live_gt_multi_feature_sources():
    ds = build_cued_bias_dataset(n_items=40, seed=2, holdout_feature="sycophantic_agreement")
    runtime = MagicMock()
    runtime.tokenizer = MagicMock()
    runtime.tokenizer.chat_template = None

    # Cycle answers so hint flips, sycophancy flips, and format parses vary.
    seq = [
        "A",
        "B",  # hint pair flip
        "C",
        "D",  # syc pair flip
        "ANSWER: A",  # format ok
        "B",
        "B",  # hint no flip
        "A",
        "A",  # syc no flip
        "not formatted",  # format fail
    ]
    answers = iter(seq * 40)

    def _gen(runtime, prompt, max_new_tokens=4, temperature=0.0):  # noqa: ANN001
        return next(answers)

    with patch(
        "introspect.introspect.model_runtime.generate_text", side_effect=_gen
    ), patch(
        "introspect.introspect.model_runtime.format_chat",
        side_effect=lambda tok, user, system=None: user,
    ):
        out = apply_live_behavioral_gt(ds, runtime, max_items=30)

    by_f = out["behavioral_gt_by_feature"]
    assert is_live_gt_source(by_f["hint_reliance"])
    assert is_live_gt_source(by_f["sycophantic_agreement"])
    assert is_live_gt_source(by_f["format_compliance"])
    assert by_f["hint_reliance"].startswith("live_")
    assert by_f["sycophantic_agreement"] == "live_opinion_flip"
    assert by_f["format_compliance"] == "live_format_parse"
    assert out["behavioral_gt_source"].startswith("live_")
    assert out["holdout_generalization_claim_ok"] is True
    assert out["holdout_behavioral_gt_source"] == "live_opinion_flip"
    # At least two live features beyond planted-only answer_certainty/planted_topic.
    live_feats = [f for f, s in by_f.items() if is_live_gt_source(s)]
    assert "hint_reliance" in live_feats
    assert len([f for f in live_feats if f != "hint_reliance"]) >= 2


def test_holdout_claim_gated_without_live_source():
    ds = build_cued_bias_dataset(n_items=40, seed=0, holdout_feature="planted_topic")
    assert ds["holdout_generalization_claim_ok"] is False
    assert not is_live_gt_source(ds["behavioral_gt_by_feature"]["planted_topic"])


def test_powered_honesty_requires_min_live_n():
    ds = build_cued_bias_dataset(n_items=50, seed=0)
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
    reports = synthetic_verbalize(ds["items"], accuracy=0.8, seed=0)

    call_n = {"n": 0}

    def fake_live(*_a, **_k):
        call_n["n"] += 1
        # Alternate so patched moves relative to null.
        return {"base": 0.0, "patched": 1.0 if call_n["n"] % 2 else 0.2, "null": 0.1}

    with patch("introspect.introspect.causal._live_report_under_patch", side_effect=fake_live):
        under = paired_activation_patch(
            acts,
            probe,
            layer=2,
            seed=0,
            reports=reports,
            runtime=object(),
            prompts=ds["items"],
            honesty_min_live_n=8,
            patch_positions=["last_token"],
            claim_patch_robustness=False,
        )
    # Budget may reach min; if it does and margin clears, pass — else underpowered.
    assert under["honesty_min_live_n"] == 8
    if under["live_regen_n"] < 8:
        assert under["passes_honesty_claim"] is False
        assert under["honesty_claim_status"] == "underpowered_live_regen"


def test_patch_robustness_fail_closed_single_position():
    ds = build_cued_bias_dataset(n_items=50, seed=1)
    acts = synthetic_activations(ds["items"], layers=[2], dim=16, seed=1)
    probes = fit_feature_probes(
        acts,
        features=["hint_reliance"],
        layer=2,
        holdout_feature="planted_topic",
        seed=1,
        min_auroc=0.0,
    )
    probe = next(iter(probes["probes"].values()))
    reports = synthetic_verbalize(ds["items"], accuracy=0.85, seed=1)

    def fake_live(*_a, **kwargs):
        return {"base": 0.0, "patched": 1.0, "null": 0.0, "patch_position": kwargs.get("patch_position")}

    with patch("introspect.introspect.causal._live_report_under_patch", side_effect=fake_live):
        out = paired_activation_patch(
            acts,
            probe,
            layer=2,
            seed=1,
            reports=reports,
            runtime=object(),
            prompts=ds["items"],
            honesty_min_live_n=4,
            patch_positions=["last_token"],  # only one tried
            claim_patch_robustness=True,
        )
    assert out["passes_honesty_claim"] is False
    assert out["honesty_claim_status"] == "patch_robustness_unproven"
    assert out["claims_patch_robustness"] is False
    assert out["patch_robustness_ok"] is False


def _strong_probe_bundle(n: int = 40, dim: int = 16, seed: int = 0):
    """Plant activations so probe causal sensitivity clearly beats shuffled null."""
    rng = np.random.default_rng(seed)
    items = []
    acts = []
    for i in range(n):
        gt = int(i % 2)
        feat = "hint_reliance"
        vec = rng.normal(0, 0.05, size=dim)
        vec[0] = 3.0 if gt == 1 else -3.0
        items.append(
            {
                "item_id": f"h_{i}",
                "feature": feat,
                "split": "train" if i < n - 8 else "test",
                "behavioral_gt": gt,
                "hint_present": True,
                "prompt": f"q{i}",
            }
        )
        acts.append(vec.tolist())
    bundle = {
        "activations": {"2": acts},
        "meta": items,
        "layers": [2],
        "dim": dim,
        "mode": "synthetic",
        "is_synthetic": True,
    }
    probe = {
        "coef": [[1.0] + [0.0] * (dim - 1)],
        "intercept": 0.0,
        "layer": 2,
    }
    return items, bundle, probe


def test_patch_position_ablation_stamps_carrier():
    items, acts, probe = _strong_probe_bundle(seed=3)
    reports = synthetic_verbalize(items, accuracy=0.9, seed=3)

    def fake_live(*_a, **kwargs):
        pos = kwargs.get("patch_position", "last_token")
        if pos == "last_token":
            return {"base": 0.0, "patched": 1.0, "null": 0.0}
        # mean_pool does not beat null
        return {"base": 0.0, "patched": 0.05, "null": 0.04}

    with patch("introspect.introspect.causal._live_report_under_patch", side_effect=fake_live):
        out = paired_activation_patch(
            acts,
            probe,
            layer=2,
            seed=3,
            reports=reports,
            runtime=object(),
            prompts=items,
            honesty_min_live_n=4,
            patch_positions=["last_token", "mean_pool"],
            claim_patch_robustness=True,
        )
    assert set(out["patch_positions_tried"]) == {"last_token", "mean_pool"}
    assert out["passes_causal_check"] is True
    assert out["patch_position_results"]["last_token"]["carries_honesty"] is True
    assert out["honesty_carrier_position"] == "last_token"
    assert out["claims_patch_robustness"] is True
    assert out["passes_honesty_claim"] is True
    assert out["honesty_claim_status"] == "live_regen"


def test_seed_agreement_stamp():
    items, acts, probe = _strong_probe_bundle(seed=4)
    reports = synthetic_verbalize(items, accuracy=0.9, seed=4)

    def fake_live(*_a, **_k):
        return {"base": 0.0, "patched": 1.0, "null": 0.0}

    with patch("introspect.introspect.causal._live_report_under_patch", side_effect=fake_live):
        out = paired_activation_patch(
            acts,
            probe,
            layer=2,
            seed=4,
            reports=reports,
            runtime=object(),
            prompts=items,
            honesty_min_live_n=4,
            honesty_seeds=[4, 5],
            patch_positions=["last_token", "mean_pool"],
            claim_patch_robustness=True,
        )
    assert out["honesty_seed_agreement"] is True
    assert out["honesty_seeds"] == [4, 5]
    assert out["passes_honesty_claim"] is True


def test_apply_residual_mean_pool_vs_last_token():
    import torch

    from introspect.introspect.causal import _apply_residual_patch

    h = torch.zeros(1, 4, 8)
    v = torch.ones(8)
    last = _apply_residual_patch(h, v, patch_position="last_token")
    assert float(last[0, -1].sum()) == 8.0
    assert float(last[0, 0].sum()) == 0.0
    mean = _apply_residual_patch(h, v, patch_position="mean_pool")
    assert float(mean[0, 0].sum()) == 8.0
    assert float(mean[0, 2].sum()) == 8.0


def test_config_honesty_min_live_n_on_eval():
    from introspect.configs.schema import EvalConfig

    ev = EvalConfig(honesty_min_live_n=8, honesty_n_seeds=2, bootstrap_samples=200)
    assert ev.honesty_min_live_n == 8
    assert ev.honesty_n_seeds == 2
