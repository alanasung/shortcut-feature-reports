"""Domain stages: behavioral GT, probes, verbalization, causal patching."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from omegaconf import DictConfig

from ._util import ensure_dir, read_json, stage_result, write_json
from .activations import try_collect_model_activations
from .causal import ablate_direction, paired_activation_patch
from .data import (
    FEATURES,
    apply_live_behavioral_gt,
    build_cued_bias_dataset,
    is_live_gt_source,
)
from .probes import fit_feature_probes
from .train import train_verbalizer
from .verbalize import score_verbalization, synthetic_verbalize

logger = logging.getLogger(__name__)


def _seed(cfg: DictConfig) -> int:
    return int(getattr(cfg.run, "seed", 0))


def _n(cfg: DictConfig) -> int:
    return max(40, int(getattr(cfg.data, "n_items", 256)))


def _layers(cfg: DictConfig) -> list[int]:
    layers = list(getattr(getattr(cfg, "eval", object()), "layers", [2, 4, 6]))
    return [int(x) for x in layers]


def _force_synthetic(cfg: Any) -> bool:
    """Synthetic is smoke-only. Pilot/full try measured weights by default."""
    if bool(getattr(cfg, "force_synthetic", False)):
        return True
    # Legacy: data=synthetic alone is not enough to force if pilot requested measured.
    exp_name = str(getattr(getattr(cfg, "experiment", object()), "name", "")).lower()
    if exp_name == "smoke":
        return True
    return False


def _revision(cfg: Any) -> str | None:
    rev = getattr(getattr(cfg, "model", cfg), "revision", None)
    return str(rev) if rev else None


def _honesty_min_live_n(cfg: Any) -> int:
    ev = getattr(cfg, "eval", None)
    if ev is not None and getattr(ev, "honesty_min_live_n", None) is not None:
        return max(1, int(ev.honesty_min_live_n))
    exp = getattr(cfg, "experiment", None)
    if exp is not None and getattr(exp, "honesty_min_live_n", None) is not None:
        return max(1, int(exp.honesty_min_live_n))
    return 8


def _honesty_seeds(cfg: Any) -> list[int]:
    base = _seed(cfg)
    ev = getattr(cfg, "eval", None)
    n = int(getattr(ev, "honesty_n_seeds", 1) or 1) if ev is not None else 1
    n = max(1, min(n, 5))
    return [base + i for i in range(n)]


def stage_build_dataset(cfg: DictConfig, run_dir: Path) -> dict[str, Any]:
    holdout = str(getattr(getattr(cfg, "experiment", object()), "holdout_feature", "sycophantic_agreement"))
    if holdout not in FEATURES:
        holdout = "sycophantic_agreement"
    ds = build_cued_bias_dataset(n_items=_n(cfg), seed=_seed(cfg), holdout_feature=holdout)
    out = ensure_dir(run_dir / "artifacts" / "dataset")
    write_json(out / "dataset.json", ds)
    metrics = {
        "n_features": len(FEATURES),
        "holdout_feature": holdout,
        "holdout_is_feature": True,
        "n_train": sum(1 for r in ds["items"] if r["split"] == "train"),
        "n_test": sum(1 for r in ds["items"] if r["split"] == "test"),
        "force_synthetic": _force_synthetic(cfg),
        "behavioral_gt_source": ds.get("behavioral_gt_source", "planted"),
        "holdout_generalization_claim_ok": bool(ds.get("holdout_generalization_claim_ok", False)),
        "honesty_min_live_n": _honesty_min_live_n(cfg),
    }
    payload = stage_result(task="build_dataset", seed=_seed(cfg), n=ds["n"], metrics=metrics)
    write_json(out / "results.json", payload)
    return payload


def stage_collect(cfg: DictConfig, run_dir: Path) -> dict[str, Any]:
    ds_path = run_dir / "artifacts" / "dataset" / "dataset.json"
    ds = read_json(ds_path) if ds_path.is_file() else build_cued_bias_dataset(n_items=_n(cfg), seed=_seed(cfg))
    model_name = str(getattr(cfg.model, "name", "sshleifer/tiny-gpt2"))
    force = _force_synthetic(cfg)
    gt_source = str(ds.get("behavioral_gt_source", "planted"))
    if not force:
        from .model_runtime import try_load_causal_lm

        runtime = try_load_causal_lm(
            model_name, revision=_revision(cfg), force_synthetic=False
        )
        if runtime is not None:
            # Cap for M4 pilot; live GT for hint / sycophancy / format features.
            ds = apply_live_behavioral_gt(
                ds, runtime, max_items=min(96, len(ds["items"]))
            )
            gt_source = str(ds.get("behavioral_gt_source", "live_multi"))
            if ds_path.parent.is_dir():
                write_json(ds_path, ds)
        elif not force:
            # Measured path without weights: fail closed (collect will raise below).
            gt_source = "planted_pending_measured_load"
    try:
        bundle = try_collect_model_activations(
            ds["items"],
            model_name=model_name,
            layers=_layers(cfg),
            seed=_seed(cfg),
            force_synthetic=force,
            revision=_revision(cfg),
            use_chat_template=bool(getattr(cfg.model, "use_chat_template", True)),
        )
    except RuntimeError as exc:
        if force:
            from .activations import synthetic_activations

            bundle = synthetic_activations(ds["items"], layers=_layers(cfg), seed=_seed(cfg))
            bundle["fallback_reason"] = str(exc)
        else:
            raise
    out = ensure_dir(run_dir / "artifacts" / "collect")
    write_json(out / "activations.json", bundle)
    metrics = {
        "mode": bundle.get("mode"),
        "dim": bundle.get("dim"),
        "layers": bundle.get("layers"),
        "revision": bundle.get("revision"),
        "chat_template_path": bundle.get("chat_template_path"),
        "fallback_reason": bundle.get("fallback_reason", ""),
        "force_synthetic": force,
        "behavioral_gt_source": gt_source,
        "behavioral_gt_by_feature": ds.get("behavioral_gt_by_feature"),
        "holdout_generalization_claim_ok": bool(ds.get("holdout_generalization_claim_ok", False)),
        "holdout_behavioral_gt_source": ds.get("holdout_behavioral_gt_source"),
    }
    payload = stage_result(task="collect", seed=_seed(cfg), n=len(bundle["meta"]), metrics=metrics)
    payload["is_synthetic"] = bool(bundle.get("is_synthetic", bundle.get("mode") != "model"))
    write_json(out / "results.json", payload)
    return payload


def stage_fit(cfg: DictConfig, run_dir: Path) -> dict[str, Any]:
    ds = read_json(run_dir / "artifacts" / "dataset" / "dataset.json")
    acts = read_json(run_dir / "artifacts" / "collect" / "activations.json")
    layer = int(_layers(cfg)[len(_layers(cfg)) // 2])
    probes = fit_feature_probes(
        acts,
        features=list(FEATURES),
        layer=layer,
        holdout_feature=ds["holdout_feature"],
        seed=_seed(cfg),
        min_auroc=0.55,
    )
    force = _force_synthetic(cfg)
    trained = train_verbalizer(
        ds,
        probes,
        acts,
        seed=_seed(cfg),
        holdout_feature=ds["holdout_feature"],
        model_name=str(getattr(cfg.model, "name", "")) or None,
        revision=_revision(cfg),
        force_synthetic=force,
    )
    out = ensure_dir(run_dir / "artifacts" / "fit")
    write_json(out / "probes.json", probes)
    write_json(
        out / "train.json",
        {
            "metrics": trained["metrics"],
            "mode": trained["mode"],
            "n_train": trained["n_train"],
            "is_synthetic": trained.get("is_synthetic", True),
            "fallback_reason": trained.get("fallback_reason", ""),
        },
    )
    write_json(out / "trained_reports.json", trained["trained_reports"])
    write_json(out / "untrained_reports.json", trained["untrained_reports"])
    metrics = {
        "n_probes": len(probes["probes"]),
        "dropped_features": probes["dropped_features"],
        "train_mode": trained["mode"],
        "is_synthetic": trained.get("is_synthetic", True),
        **trained["metrics"],
    }
    metrics["trained_seen_acc"] = trained["metrics"]["trained_seen"]["accuracy_behavioral"]
    metrics["trained_holdout_acc"] = trained["metrics"]["trained_holdout"]["accuracy_behavioral"]
    holdout_source = (
        (ds.get("behavioral_gt_by_feature") or {}).get(ds["holdout_feature"])
        or ds.get("holdout_behavioral_gt_source")
        or ds.get("behavioral_gt_source")
        or "planted"
    )
    holdout_claim_ok = is_live_gt_source(str(holdout_source))
    metrics["holdout_behavioral_gt_source"] = holdout_source
    metrics["holdout_generalization_claim_ok"] = holdout_claim_ok
    if not holdout_claim_ok:
        # Gap is still recorded as a diagnostic; claims must not treat it as evidence.
        metrics["holdout_generalization_gap_claim"] = None
        metrics["holdout_generalization_note"] = (
            "Holdout generalization claim gated off: holdout feature lacks live_* "
            f"behavioral_gt_source (got {holdout_source!r})"
        )
    else:
        metrics["holdout_generalization_gap_claim"] = metrics.get("holdout_generalization_gap")
    payload = stage_result(task="fit", seed=_seed(cfg), n=trained["n_train"], metrics=metrics)
    payload["is_synthetic"] = bool(trained.get("is_synthetic", True))
    write_json(out / "results.json", payload)
    return payload


def stage_evaluate(cfg: DictConfig, run_dir: Path) -> dict[str, Any]:
    ds = read_json(run_dir / "artifacts" / "dataset" / "dataset.json")
    acts = read_json(run_dir / "artifacts" / "collect" / "activations.json")
    probes = read_json(run_dir / "artifacts" / "fit" / "probes.json")
    trained_reports = read_json(run_dir / "artifacts" / "fit" / "trained_reports.json")
    fit_meta = {}
    fit_path = run_dir / "artifacts" / "fit" / "train.json"
    if fit_path.is_file():
        fit_meta = read_json(fit_path)
    force = _force_synthetic(cfg)
    # Synthetic shortcut baselines are always stamped; excluded from headline deltas
    # when a measured claim is asserted.
    baselines = {}
    for i, name in enumerate(("introspective", "input_only", "metadata_only", "shuffled")):
        scored = score_verbalization(
            synthetic_verbalize(ds["items"], seed=_seed(cfg) + i, baseline=name)
        )
        scored["is_synthetic"] = True
        baselines[name] = scored
    causal = {"passes_causal_check": False, "passes_honesty_claim": False}
    ablation = {}
    runtime = None
    if not force:
        try:
            from .model_runtime import try_load_causal_lm

            runtime = try_load_causal_lm(
                str(getattr(cfg.model, "name", "")),
                revision=_revision(cfg),
                force_synthetic=False,
            )
        except Exception as exc:  # noqa: BLE001
            runtime = None
            causal = {
                "passes_causal_check": False,
                "passes_honesty_claim": False,
                "honesty_claim_status": "live_regen_failed",
                "live_regen_errors": [str(exc)],
            }
    min_live_n = _honesty_min_live_n(cfg)
    h_seeds = _honesty_seeds(cfg)
    for feat, probe in probes["probes"].items():
        if feat == ds["holdout_feature"]:
            continue
        causal = paired_activation_patch(
            acts,
            probe,
            layer=int(probes["layer"]),
            seed=_seed(cfg),
            reports=trained_reports,
            runtime=runtime,
            prompts=ds["items"],
            honesty_min_live_n=min_live_n,
            honesty_seeds=h_seeds,
            claim_patch_robustness=True,
        )
        ablation = ablate_direction(acts, probe, layer=int(probes["layer"]))
        break
    test_reports = [r for r in trained_reports if r.get("split") == "test"] or trained_reports
    headline = score_verbalization(test_reports)
    measured_claims = (not force) and (not bool(fit_meta.get("is_synthetic", True)))
    # When measured claims are asserted, synthetic baselines must not drive deltas.
    baseline_delta = None
    untrained_acc = None
    if measured_claims:
        untrained_path = run_dir / "artifacts" / "fit" / "untrained_reports.json"
        if untrained_path.is_file():
            untrained_acc = score_verbalization(read_json(untrained_path))["accuracy_behavioral"]
            baseline_delta = headline["accuracy_behavioral"] - untrained_acc
    holdout_source = (
        (ds.get("behavioral_gt_by_feature") or {}).get(ds["holdout_feature"])
        or ds.get("holdout_behavioral_gt_source")
        or ds.get("behavioral_gt_source")
        or "planted"
    )
    metrics = {
        "accuracy_behavioral": headline["accuracy_behavioral"],
        "ece": headline["ece"],
        "parse_coverage": headline.get("parse_coverage"),
        "n_eval": headline.get("n"),
        "eval_split": "test",
        "baselines": baselines,
        "baselines_are_synthetic": True,
        "exclude_synthetic_baselines_from_headline_deltas": bool(measured_claims),
        "untrained_accuracy_behavioral": untrained_acc,
        "trained_minus_untrained": baseline_delta,
        "causal": causal,
        "ablation": ablation,
        "probe_agreement_is_not_evidence": True,
        "behavioral_gt_source": ds.get("behavioral_gt_source", "planted"),
        "behavioral_gt_by_feature": ds.get("behavioral_gt_by_feature"),
        "holdout_behavioral_gt_source": holdout_source,
        "holdout_generalization_claim_ok": is_live_gt_source(str(holdout_source)),
        "honesty_min_live_n": min_live_n,
        "note": "headline metric is behavioral GT accuracy on locked test split, not probe agreement",
        "honesty_requires_live_regen": True,
        "measured_claims_ok": bool(measured_claims)
        and causal.get("honesty_claim_status")
        not in {"live_regen_failed", "underpowered_live_regen", "patch_robustness_unproven"},
    }
    out = ensure_dir(run_dir / "artifacts" / "evaluate")
    payload = stage_result(task="evaluate", seed=_seed(cfg), n=len(trained_reports), metrics=metrics)
    payload["is_synthetic"] = bool(force or fit_meta.get("is_synthetic", False))
    write_json(out / "results.json", payload)
    return payload


def stage_report(cfg: DictConfig, run_dir: Path) -> dict[str, Any]:
    pieces = {}
    for name in ("build_dataset", "collect", "fit", "evaluate"):
        p = run_dir / "artifacts" / name / "results.json"
        if p.is_file():
            pieces[name] = read_json(p)
    metrics = {
        "accuracy_behavioral": pieces.get("evaluate", {}).get("metrics", {}).get("accuracy_behavioral"),
        "holdout_generalization_gap": pieces.get("fit", {}).get("metrics", {}).get("holdout_generalization_gap"),
        "causal_sensitivity": pieces.get("evaluate", {}).get("metrics", {}).get("causal", {}).get("causal_sensitivity"),
        "passes_causal_check": pieces.get("evaluate", {}).get("metrics", {}).get("causal", {}).get("passes_causal_check"),
        "passes_honesty_claim": pieces.get("evaluate", {}).get("metrics", {}).get("causal", {}).get("passes_honesty_claim"),
        "honesty_carrier_position": pieces.get("evaluate", {}).get("metrics", {}).get("causal", {}).get(
            "honesty_carrier_position"
        ),
        "holdout_generalization_claim_ok": pieces.get("fit", {}).get("metrics", {}).get(
            "holdout_generalization_claim_ok"
        ),
        "collect_mode": pieces.get("collect", {}).get("metrics", {}).get("mode"),
        "train_mode": pieces.get("fit", {}).get("metrics", {}).get("train_mode"),
    }
    out = ensure_dir(run_dir / "artifacts" / "report")
    write_json(out / "summary.json", {"stages": pieces, "metrics": metrics})
    payload = stage_result(task="report", seed=_seed(cfg), n=len(pieces), metrics=metrics)
    write_json(out / "results.json", payload)
    return payload
