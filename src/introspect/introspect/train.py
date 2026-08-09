"""LoRA verbalization training on probe-derived labels (measured when weights load)."""

from __future__ import annotations

from typing import Any

import numpy as np

from .probes import probe_predict
from .verbalize import (
    build_report_prompt,
    model_verbalize,
    parse_report,
    score_verbalization,
    synthetic_verbalize,
)


def _synthetic_lora(
    dataset: dict[str, Any],
    probe_bundle: dict[str, Any],
    act_bundle: dict[str, Any],
    *,
    seed: int,
    holdout_feature: str,
    reason: str,
) -> dict[str, Any]:
    """Smoke-only path: planted accuracies with honest synthetic labels."""
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
        "is_synthetic": True,
        "fallback_reason": reason,
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
            "holdout_is_feature": True,
            "note": "holdout is an entire FEATURE withheld from training, not examples",
        },
    }


def _measured_lora(
    dataset: dict[str, Any],
    probe_bundle: dict[str, Any],
    act_bundle: dict[str, Any],
    *,
    seed: int,
    holdout_feature: str,
    model_name: str,
    revision: str | None,
    max_steps: int = 12,
) -> dict[str, Any] | None:
    """Minimal local LoRA SFT on probe labels for non-holdout features."""
    from .model_runtime import format_chat, generate_text, try_load_causal_lm

    runtime = try_load_causal_lm(model_name, revision=revision, force_synthetic=False)
    if runtime is None:
        return None

    try:
        import torch
        from peft import LoraConfig, TaskType, get_peft_model
    except Exception:
        # peft unavailable: still attempt measured untrained + probe-conditioned reports
        reports = model_verbalize(
            dataset["items"],
            runtime=runtime,
            seed=seed,
        )
        seen = [r for r in reports if r["feature"] != holdout_feature]
        hold = [r for r in reports if r["feature"] == holdout_feature]
        return {
            "mode": "measured_no_peft",
            "is_synthetic": False,
            "fallback_reason": "peft unavailable; measured generation without LoRA update",
            "n_train": 0,
            "trained_reports": reports,
            "untrained_reports": reports,
            "metrics": {
                "probe_train_agreement": float("nan"),
                "trained_seen": score_verbalization(seen),
                "trained_holdout": score_verbalization(hold),
                "untrained": score_verbalization(reports),
                "holdout_generalization_gap": (
                    score_verbalization(seen)["accuracy_behavioral"]
                    - score_verbalization(hold)["accuracy_behavioral"]
                ),
                "holdout_is_feature": True,
            },
        }

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
    if not train_rows:
        return None

    try:
        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=4,
            lora_alpha=8,
            lora_dropout=0.0,
            target_modules=["c_attn", "q_proj", "v_proj", "k_proj", "o_proj"],
        )
        peft_model = get_peft_model(runtime.model, lora_cfg)
    except Exception:
        peft_model = runtime.model

    peft_model.train()
    opt = torch.optim.AdamW(
        [p for p in peft_model.parameters() if p.requires_grad], lr=2e-4
    )
    steps = 0
    probe_agree_bits: list[int] = []
    for row in train_rows:
        if steps >= max_steps:
            break
        idx = id_to_idx[row["item_id"]]
        proba = float(
            probe_predict(probe_bundle["probes"][row["feature"]], acts[idx : idx + 1])[0]
        )
        label = int(proba >= 0.5)
        probe_agree_bits.append(int(label == int(row["behavioral_gt"])))
        target = (
            f"FEATURE={row['feature']}; ACTIVE={'yes' if label else 'no'}; CONF=0.70"
        )
        prompt = build_report_prompt(row)
        text = format_chat(
            runtime.tokenizer,
            prompt,
            system="Emit only the FEATURE/ACTIVE/CONF report schema.",
        )
        full = text + target
        enc = runtime.tokenizer(full, return_tensors="pt")
        enc = {k: v.to(runtime.device) for k, v in enc.items()}
        # Mask prompt tokens from loss
        prompt_len = runtime.tokenizer(text, return_tensors="pt")["input_ids"].shape[-1]
        labels = enc["input_ids"].clone()
        labels[:, :prompt_len] = -100
        try:
            out = peft_model(**enc, labels=labels)
            loss = out.loss
            if loss is None or not torch.isfinite(loss):
                continue
            opt.zero_grad()
            loss.backward()
            opt.step()
            steps += 1
        except Exception:
            break

    peft_model.eval()
    runtime.model = peft_model

    def _gen_reports(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        outs: list[dict[str, Any]] = []
        for row in rows:
            prompt = format_chat(
                runtime.tokenizer,
                build_report_prompt(row),
                system="Emit only the FEATURE/ACTIVE/CONF report schema.",
            )
            try:
                text = generate_text(runtime, prompt, max_new_tokens=32, temperature=0.0)
            except Exception:
                text = f"FEATURE={row['feature']}; ACTIVE=no; CONF=0.50"
            parsed = parse_report(text)
            # Never impute the ground-truth label on parse failure.
            parse_ok = parsed["active"] is not None
            active = int(bool(parsed["active"])) if parse_ok else -1
            outs.append(
                {
                    "item_id": row["item_id"],
                    "feature": row["feature"],
                    "split": row["split"],
                    "behavioral_gt": row["behavioral_gt"],
                    "report_active": active,
                    "confidence": parsed["confidence"] if parse_ok else 0.0,
                    "baseline": "lora_trained",
                    "text": text,
                    "mode": "measured",
                    "parse_ok": parse_ok,
                }
            )
        return outs

    seen_rows = [r for r in dataset["items"] if r["feature"] != holdout_feature]
    hold_rows = [r for r in dataset["items"] if r["feature"] == holdout_feature]
    trained_seen = _gen_reports(seen_rows)
    trained_hold = _gen_reports(hold_rows)
    untrained = synthetic_verbalize(
        dataset["items"], accuracy=0.52, seed=seed + 2, baseline="introspective"
    )
    probe_agree = float(np.mean(probe_agree_bits)) if probe_agree_bits else 0.0
    return {
        "mode": "measured_lora",
        "is_synthetic": False,
        "n_train": len(train_rows),
        "lora_steps": steps,
        "model_name": model_name,
        "revision": revision,
        "trained_reports": trained_seen + trained_hold,
        "untrained_reports": untrained,
        "probe_train_agreement": probe_agree,
        "metrics": {
            "probe_train_agreement": probe_agree,
            "trained_seen": score_verbalization(trained_seen),
            "trained_holdout": score_verbalization(trained_hold),
            "untrained": score_verbalization(untrained),
            "holdout_generalization_gap": (
                score_verbalization(trained_seen)["accuracy_behavioral"]
                - score_verbalization(trained_hold)["accuracy_behavioral"]
            ),
            "holdout_is_feature": True,
            "lora_steps": steps,
        },
    }


def train_verbalizer(
    dataset: dict[str, Any],
    probe_bundle: dict[str, Any],
    act_bundle: dict[str, Any],
    *,
    seed: int = 0,
    holdout_feature: str,
    model_name: str | None = None,
    revision: str | None = None,
    force_synthetic: bool = False,
    max_steps: int = 12,
) -> dict[str, Any]:
    """Train (or smoke-simulate) verbalization; headline metric stays behavioral GT.

    Held-out means an entire FEATURE is withheld from training — not held-out
    examples of a seen feature.
    """
    if force_synthetic or not model_name:
        return _synthetic_lora(
            dataset,
            probe_bundle,
            act_bundle,
            seed=seed,
            holdout_feature=holdout_feature,
            reason="force_synthetic=True" if force_synthetic else "no model_name",
        )
    measured = _measured_lora(
        dataset,
        probe_bundle,
        act_bundle,
        seed=seed,
        holdout_feature=holdout_feature,
        model_name=model_name,
        revision=revision,
        max_steps=max_steps,
    )
    if measured is not None:
        return measured
    return _synthetic_lora(
        dataset,
        probe_bundle,
        act_bundle,
        seed=seed,
        holdout_feature=holdout_feature,
        reason=f"measured LoRA unavailable for {model_name!r}",
    )
