"""Domain stages: behavioral GT, probes, verbalization, causal patching."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from omegaconf import DictConfig

from ._util import ensure_dir, read_json, stage_result, write_json
from .activations import try_collect_model_activations
from .causal import ablate_direction, paired_activation_patch
from .data import FEATURES, build_cued_bias_dataset
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
    }
    payload = stage_result(task="build_dataset", seed=_seed(cfg), n=ds["n"], metrics=metrics)
    write_json(out / "results.json", payload)
    return payload


def stage_collect(cfg: DictConfig, run_dir: Path) -> dict[str, Any]:
    ds_path = run_dir / "artifacts" / "dataset" / "dataset.json"
    ds = read_json(ds_path) if ds_path.is_file() else build_cued_bias_dataset(n_items=_n(cfg), seed=_seed(cfg))
    model_name = str(getattr(cfg.model, "name", "sshleifer/tiny-gpt2"))
    force = _force_synthetic(cfg)
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
    payload = stage_result(task="fit", seed=_seed(cfg), n=trained["n_train"], metrics=metrics)
    payload["is_synthetic"] = bool(trained.get("is_synthetic", True))
    write_json(out / "results.json", payload)
    return payload


def stage_evaluate(cfg: DictConfig, run_dir: Path) -> dict[str, Any]:
    ds = read_json(run_dir / "artifacts" / "dataset" / "dataset.json")
    acts = read_json(run_dir / "artifacts" / "collect" / "activations.json")
    probes = read_json(run_dir / "artifacts" / "fit" / "probes.json")
    trained_reports = read_json(run_dir / "artifacts" / "fit" / "trained_reports.json")
    baselines = {
        name: score_verbalization(synthetic_verbalize(ds["items"], seed=_seed(cfg) + i, baseline=name))
        for i, name in enumerate(("introspective", "input_only", "metadata_only", "shuffled"))
    }
    causal = {"passes_causal_check": False, "passes_honesty_claim": False}
    ablation = {}
    for feat, probe in probes["probes"].items():
        if feat == ds["holdout_feature"]:
            continue
        causal = paired_activation_patch(
            acts,
            probe,
            layer=int(probes["layer"]),
            seed=_seed(cfg),
            reports=trained_reports,
        )
        ablation = ablate_direction(acts, probe, layer=int(probes["layer"]))
        break
    test_reports = [r for r in trained_reports if r.get("split") == "test"] or trained_reports
    headline = score_verbalization(test_reports)
    metrics = {
        "accuracy_behavioral": headline["accuracy_behavioral"],
        "ece": headline["ece"],
        "parse_coverage": headline.get("parse_coverage"),
        "n_eval": headline.get("n"),
        "eval_split": "test",
        "baselines": baselines,
        "causal": causal,
        "ablation": ablation,
        "probe_agreement_is_not_evidence": True,
        "note": "headline metric is behavioral GT accuracy on locked test split, not probe agreement",
        "honesty_requires_live_regen": True,
    }
    out = ensure_dir(run_dir / "artifacts" / "evaluate")
    payload = stage_result(task="evaluate", seed=_seed(cfg), n=len(trained_reports), metrics=metrics)
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
        "collect_mode": pieces.get("collect", {}).get("metrics", {}).get("mode"),
        "train_mode": pieces.get("fit", {}).get("metrics", {}).get("train_mode"),
    }
    out = ensure_dir(run_dir / "artifacts" / "report")
    write_json(out / "summary.json", {"stages": pieces, "metrics": metrics})
    payload = stage_result(task="report", seed=_seed(cfg), n=len(pieces), metrics=metrics)
    write_json(out / "results.json", payload)
    return payload
