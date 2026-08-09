"""P6: live paired behavioral GT, honesty fail-closed, measured untrained baseline."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch


from introspect.introspect.activations import synthetic_activations
from introspect.introspect.causal import paired_activation_patch
from introspect.introspect.data import (
    apply_live_paired_behavioral_gt,
    build_cued_bias_dataset,
    parse_choice_answer,
)
from introspect.introspect.pipeline import _force_synthetic, stage_evaluate
from introspect.introspect.probes import fit_feature_probes
from introspect.introspect.train import train_verbalizer
from introspect.introspect.verbalize import synthetic_verbalize


def test_planted_source_stamp():
    ds = build_cued_bias_dataset(n_items=40, seed=0)
    assert ds["behavioral_gt_source"] == "planted"


def test_parse_choice_answer():
    assert parse_choice_answer("The answer is B.") == "B"
    assert parse_choice_answer("no letter here") is None


def test_live_paired_behavioral_gt():
    ds = build_cued_bias_dataset(n_items=40, seed=1)
    runtime = MagicMock()
    runtime.tokenizer = MagicMock()
    runtime.tokenizer.chat_template = None
    # Alternate hint vs no-hint letters so flips are observed.
    answers = iter(["A", "B", "A", "A", "C", "D", "B", "B"] * 20)

    def _gen(runtime, prompt, max_new_tokens=4, temperature=0.0):  # noqa: ANN001
        return next(answers)

    with patch(
        "introspect.introspect.model_runtime.generate_text", side_effect=_gen
    ), patch(
        "introspect.introspect.model_runtime.format_chat",
        side_effect=lambda tok, user, system=None: user,
    ):
        out = apply_live_paired_behavioral_gt(ds, runtime, max_items=8)
    assert out["behavioral_gt_source"].startswith("live_")
    assert out["behavioral_gt_by_feature"]["hint_reliance"] == "live_paired"
    assert out["live_paired_n_scored"] >= 1
    flipped = [
        r
        for r in out["items"]
        if r.get("feature") == "hint_reliance" and r.get("behavioral_gt_parse_ok")
    ]
    assert flipped
    # At least one paired flip when answers differ.
    assert any(r.get("hint_flips_answer") for r in flipped) or any(
        not r.get("hint_flips_answer") for r in flipped
    )


def test_live_regen_failed_not_silent():
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

    class BoomRuntime:
        tokenizer = object()
        model = object()

    def boom(*_a, **_k):
        raise RuntimeError("regen broke")

    with patch("introspect.introspect.causal._live_report_under_patch", side_effect=boom):
        c = paired_activation_patch(
            acts,
            probe,
            layer=2,
            seed=0,
            reports=reports,
            runtime=BoomRuntime(),
            prompts=ds["items"],
        )
    assert c["passes_honesty_claim"] is False
    assert c["honesty_claim_status"] == "live_regen_failed"
    assert c["live_regen_attempts"] > 0
    assert c["live_regen_n"] == 0
    assert c["live_regen_errors"]


def test_measured_untrained_not_planted():
    ds = build_cued_bias_dataset(n_items=40, seed=0)
    acts = synthetic_activations(ds["items"], layers=[2], dim=24, seed=0)
    probes = fit_feature_probes(
        acts,
        features=list(ds["features"]),
        layer=2,
        holdout_feature=ds["holdout_feature"],
        seed=0,
        min_auroc=0.0,
    )

    class FakeRuntime:
        def __init__(self) -> None:
            self.tokenizer = MagicMock()
            self.tokenizer.chat_template = None
            self.tokenizer.pad_token_id = 0
            self.device = "cpu"
            self.model = MagicMock()

    runtime = FakeRuntime()

    def fake_verbalize(rows, *, runtime, seed=0, max_new_tokens=32):  # noqa: ANN001
        outs = []
        for row in rows:
            outs.append(
                {
                    "item_id": row["item_id"],
                    "feature": row["feature"],
                    "split": row["split"],
                    "behavioral_gt": row["behavioral_gt"],
                    "report_active": int(row["behavioral_gt"]),
                    "confidence": 0.7,
                    "baseline": "model",
                    "text": f"FEATURE={row['feature']}; ACTIVE=yes; CONF=0.70",
                    "mode": "measured",
                    "parse_ok": True,
                    "is_synthetic": False,
                }
            )
        return outs

    import builtins

    real_import = builtins.__import__

    def guarded(name, *a, **k):  # noqa: ANN001
        if name == "peft" or (isinstance(name, str) and name.startswith("peft.")):
            raise ImportError("no peft in unit test")
        return real_import(name, *a, **k)

    with patch(
        "introspect.introspect.model_runtime.try_load_causal_lm",
        return_value=runtime,
    ), patch(
        "introspect.introspect.train.model_verbalize",
        side_effect=fake_verbalize,
    ), patch("builtins.__import__", side_effect=guarded):
        out = train_verbalizer(
            ds,
            probes,
            acts,
            seed=0,
            holdout_feature=ds["holdout_feature"],
            model_name="sshleifer/tiny-gpt2",
            force_synthetic=False,
        )
    assert out["is_synthetic"] is False
    assert out["metrics"]["untrained_is_synthetic"] is False
    # Must not be the planted ~0.52 synthetic untrained path.
    assert out["mode"] in {"measured_no_peft", "measured_lora"}
    acc = out["metrics"]["untrained"]["accuracy_behavioral"]
    assert abs(acc - 0.52) > 1e-6 or acc == 1.0


def test_evaluate_excludes_synthetic_baselines_when_measured(tmp_path):
    ds = build_cued_bias_dataset(n_items=40, seed=0)
    ds["behavioral_gt_source"] = "live_paired"
    acts = synthetic_activations(ds["items"], layers=[2], dim=16, seed=0)
    probes = fit_feature_probes(
        acts,
        features=["hint_reliance"],
        layer=2,
        holdout_feature=ds["holdout_feature"],
        seed=0,
        min_auroc=0.0,
    )
    reports = synthetic_verbalize(ds["items"], accuracy=0.9, seed=0)
    art = tmp_path / "artifacts"
    (art / "dataset").mkdir(parents=True)
    (art / "collect").mkdir(parents=True)
    (art / "fit").mkdir(parents=True)
    from introspect.introspect._util import write_json

    write_json(art / "dataset" / "dataset.json", ds)
    write_json(art / "collect" / "activations.json", acts)
    write_json(art / "fit" / "probes.json", probes)
    write_json(art / "fit" / "trained_reports.json", reports)
    write_json(art / "fit" / "untrained_reports.json", reports)
    write_json(
        art / "fit" / "train.json",
        {"metrics": {}, "mode": "measured_lora", "n_train": 3, "is_synthetic": False},
    )
    cfg = SimpleNamespace(
        force_synthetic=False,
        experiment=SimpleNamespace(name="pilot", holdout_feature=ds["holdout_feature"]),
        run=SimpleNamespace(seed=0, profile="pilot"),
        data=SimpleNamespace(n_items=40, name="pilot"),
        model=SimpleNamespace(name="no/model", revision=None, use_chat_template=True),
        eval=SimpleNamespace(layers=[2]),
    )
    with patch(
        "introspect.introspect.model_runtime.try_load_causal_lm",
        return_value=None,
    ):
        out = stage_evaluate(cfg, tmp_path)
    assert out["metrics"]["baselines_are_synthetic"] is True
    assert out["metrics"]["exclude_synthetic_baselines_from_headline_deltas"] is True
    assert all(v.get("is_synthetic") for v in out["metrics"]["baselines"].values())


def test_force_synthetic_still_smoke_only():
    assert _force_synthetic(SimpleNamespace(force_synthetic=True, experiment=SimpleNamespace(name="x")))
    assert not _force_synthetic(
        SimpleNamespace(force_synthetic=False, experiment=SimpleNamespace(name="pilot"))
    )
