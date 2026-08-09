"""Domain stages: behavioral GT, probes, verbalization, causal patching."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from omegaconf import DictConfig

from .activations import try_collect_model_activations
from .causal import ablate_direction, paired_activation_patch
from .data import FEATURES, build_cued_bias_dataset
from .probes import fit_feature_probes
from .train import train_verbalizer
from .verbalize import score_verbalization, synthetic_verbalize
from ._util import ensure_dir, read_json, stage_result, write_json

logger = logging.getLogger(__name__)


def _seed(cfg: DictConfig) -> int:
    return int(getattr(cfg.run, "seed", 0))


def _n(cfg: DictConfig) -> int:
    return max(40, int(getattr(cfg.data, "n_items", 256)))


def _layers(cfg: DictConfig) -> list[int]:
    layers = list(getattr(getattr(cfg, "eval", object()), "layers", [2, 4, 6]))
    return [int(x) for x in layers]


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
        "n_train": sum(1 for r in ds["items"] if r["split"] == "train"),
        "n_test": sum(1 for r in ds["items"] if r["split"] == "test"),
    }
    payload = stage_result(task="build_dataset", seed=_seed(cfg), n=ds["n"], metrics=metrics)
    write_json(out / "results.json", payload)
    return payload


def stage_collect(cfg: DictConfig, run_dir: Path) -> dict[str, Any]:
    ds_path = run_dir / "artifacts" / "dataset" / "dataset.json"
    ds = read_json(ds_path) if ds_path.is_file() else build_cued_bias_dataset(n_items=_n(cfg), seed=_seed(cfg))
    model_name = str(getattr(cfg.model, "name", "sshleifer/tiny-gpt2"))
    force = str(getattr(cfg.data, "name", "synthetic")) == "synthetic" or str(getattr(cfg.run, "profile", "")) in {"smoke", "pilot"}
    # Pilot defaults to synthetic unless explicitly requesting measured weights.
    force = force or not bool(getattr(cfg.model, "require_weights", False))
    bundle = try_collect_model_activations(
        ds["items"],
        model_name=model_name,
        layers=_layers(cfg),
        seed=_seed(cfg),
        force_synthetic=force,
    )
    out = ensure_dir(run_dir / "artifacts" / "collect")
    write_json(out / "activations.json", bundle)
    metrics = {
        "mode": bundle.get("mode"),
        "dim": bundle.get("dim"),
        "layers": bundle.get("layers"),
        "fallback_reason": bundle.get("fallback_reason", ""),
    }
    payload = stage_result(task="collect", seed=_seed(cfg), n=len(bundle["meta"]), metrics=metrics)
    payload["is_synthetic"] = bundle.get("mode") != "model"
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
    trained = train_verbalizer(
        ds, probes, acts, seed=_seed(cfg), holdout_feature=ds["holdout_feature"]
    )
    out = ensure_dir(run_dir / "artifacts" / "fit")
    write_json(out / "probes.json", probes)
    write_json(out / "train.json", {"metrics": trained["metrics"], "mode": trained["mode"], "n_train": trained["n_train"]})
    write_json(out / "trained_reports.json", trained["trained_reports"])
    write_json(out / "untrained_reports.json", trained["untrained_reports"])
    metrics = {
        "n_probes": len(probes["probes"]),
        "dropped_features": probes["dropped_features"],
        **trained["metrics"],
    }
    # Flatten nested for reporting convenience
    metrics["trained_seen_acc"] = trained["metrics"]["trained_seen"]["accuracy_behavioral"]
    metrics["trained_holdout_acc"] = trained["metrics"]["trained_holdout"]["accuracy_behavioral"]
    payload = stage_result(task="fit", seed=_seed(cfg), n=trained["n_train"], metrics=metrics)
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
    # Causal check on first accepted non-holdout probe
    causal = {"passes_causal_check": False}
    ablation = {}
    for feat, probe in probes["probes"].items():
        if feat == ds["holdout_feature"]:
            continue
        causal = paired_activation_patch(acts, probe, layer=int(probes["layer"]), seed=_seed(cfg))
        ablation = ablate_direction(acts, probe, layer=int(probes["layer"]))
        break
    headline = score_verbalization(trained_reports)
    metrics = {
        "accuracy_behavioral": headline["accuracy_behavioral"],
        "ece": headline["ece"],
        "baselines": baselines,
        "causal": causal,
        "ablation": ablation,
        "probe_agreement_is_not_evidence": True,
        "note": "headline metric is behavioral GT accuracy, not probe agreement",
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
    }
    out = ensure_dir(run_dir / "artifacts" / "report")
    write_json(out / "summary.json", {"stages": pieces, "metrics": metrics})
    payload = stage_result(task="report", seed=_seed(cfg), n=len(pieces), metrics=metrics)
    write_json(out / "results.json", payload)
    return payload
