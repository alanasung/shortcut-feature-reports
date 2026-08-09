"""Paired activation patching and ablation causal checks."""

from __future__ import annotations

from typing import Any

import numpy as np

from .probes import probe_predict


def paired_activation_patch(
    act_bundle: dict[str, Any],
    probe: dict[str, Any],
    *,
    layer: int,
    seed: int = 0,
) -> dict[str, Any]:
    """Swap activations between matched pairs and measure probe movement."""
    rng = np.random.default_rng(seed)
    acts = np.asarray(act_bundle["activations"][str(layer)], dtype=np.float64)
    meta = act_bundle["meta"]
    by_feat: dict[str, list[int]] = {}
    for i, m in enumerate(meta):
        by_feat.setdefault(m["feature"], []).append(i)

    deltas: list[float] = []
    shuffled_deltas: list[float] = []
    for _feature, idxs in by_feat.items():
        pos = [i for i in idxs if meta[i]["behavioral_gt"] == 1]
        neg = [i for i in idxs if meta[i]["behavioral_gt"] == 0]
        n = min(len(pos), len(neg), 32)
        if n == 0:
            continue
        for k in range(n):
            i, j = pos[k], neg[k]
            base_i = float(probe_predict(probe, acts[i : i + 1])[0])
            patched = acts[j]
            after = float(probe_predict(probe, patched.reshape(1, -1))[0])
            deltas.append(after - base_i)
            r = int(rng.integers(0, len(acts)))
            null = float(probe_predict(probe, acts[r : r + 1])[0]) - base_i
            shuffled_deltas.append(null)

    sens = float(np.mean(np.abs(deltas))) if deltas else 0.0
    null = float(np.mean(np.abs(shuffled_deltas))) if shuffled_deltas else 0.0
    return {
        "layer": layer,
        "n_pairs": len(deltas),
        "causal_sensitivity": sens,
        "shuffled_null": null,
        "sensitivity_minus_null": sens - null,
        "passes_causal_check": bool(sens > null + 0.05),
    }


def ablate_direction(
    act_bundle: dict[str, Any],
    probe: dict[str, Any],
    *,
    layer: int,
) -> dict[str, Any]:
    coef = np.asarray(probe["coef"], dtype=np.float64).reshape(-1)
    direction = coef / (np.linalg.norm(coef) + 1e-8)
    acts = np.asarray(act_bundle["activations"][str(layer)], dtype=np.float64)
    before = probe_predict(probe, acts)
    ablated = acts - (acts @ direction)[:, None] * direction[None, :]
    after = probe_predict(probe, ablated)
    return {
        "mean_abs_change": float(np.mean(np.abs(after - before))),
        "mean_before": float(before.mean()),
        "mean_after": float(after.mean()),
    }
