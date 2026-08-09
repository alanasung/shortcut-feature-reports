"""LoRA-style verbalization training on probe-derived labels (synthetic path)."""

from __future__ import annotations

from typing import Any

import numpy as np

from .probes import probe_predict
from .verbalize import score_verbalization, synthetic_verbalize


def train_verbalizer(
    dataset: dict[str, Any],
    probe_bundle: dict[str, Any],
    act_bundle: dict[str, Any],
    *,
    seed: int = 0,
    holdout_feature: str,
) -> dict[str, Any]:
    """Simulate LoRA fine-tuning by increasing report→behavioral agreement.

    Training labels come from probes on non-holdout features; evaluation never
    uses probe labels as the headline metric.
    """
    layer = int(probe_bundle["layer"])
    acts = np.asarray(act_bundle["activations"][str(layer)], dtype=np.float64)
    id_to_idx = {m["item_id"]: i for i, m in enumerate(act_bundle["meta"])}

    train_rows = [
        r
        for r in dataset["items"]
        if r["split"] == "train"
        and r["feature"] != holdout_feature
        and r["feature"] in probe_bundle["probes"]
    ]
    probe_labels = []
    for row in train_rows:
        idx = id_to_idx[row["item_id"]]
        proba = float(
            probe_predict(probe_bundle["probes"][row["feature"]], acts[idx : idx + 1])[0]
        )
        probe_labels.append(int(proba >= 0.5))

    probe_agree = (
        float(
            np.mean(
                [
                    int(a == b)
                    for a, b in zip(probe_labels, [r["behavioral_gt"] for r in train_rows])
                ]
            )
        )
        if train_rows
        else 0.0
    )

    seen = [r for r in dataset["items"] if r["feature"] != holdout_feature]
    hold = [r for r in dataset["items"] if r["feature"] == holdout_feature]
    trained_seen = synthetic_verbalize(seen, accuracy=0.86, seed=seed, baseline="introspective")
    trained_hold = synthetic_verbalize(hold, accuracy=0.62, seed=seed + 1, baseline="introspective")
    untrained = synthetic_verbalize(
        dataset["items"], accuracy=0.52, seed=seed + 2, baseline="introspective"
    )

    return {
        "mode": "synthetic_lora",
        "probe_train_agreement": probe_agree,
        "n_train": len(train_rows),
        "trained_reports": trained_seen + trained_hold,
        "untrained_reports": untrained,
        "metrics": {
            "probe_train_agreement": probe_agree,
            "trained_seen": score_verbalization(trained_seen),
            "trained_holdout": score_verbalization(trained_hold),
            "untrained": score_verbalization(untrained),
            "holdout_generalization_gap": (
                score_verbalization(trained_seen)["accuracy_behavioral"]
                - score_verbalization(trained_hold)["accuracy_behavioral"]
            ),
        },
    }
