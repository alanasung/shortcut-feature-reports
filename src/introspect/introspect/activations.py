"""Residual-stream activation capture with a synthetic fallback (smoke-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from .model_runtime import format_chat, try_load_causal_lm


def synthetic_activations(
    items: list[dict[str, Any]],
    *,
    layers: list[int],
    dim: int = 64,
    seed: int = 0,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    layer_mats: dict[str, list[list[float]]] = {str(layer): [] for layer in layers}
    meta: list[dict[str, Any]] = []
    for i, row in enumerate(items):
        base = rng.normal(0.0, 1.0, size=dim)
        direction = rng.normal(0.0, 1.0, size=dim)
        direction = direction / (np.linalg.norm(direction) + 1e-8)
        signal = (2 * int(row["behavioral_gt"]) - 1) * 1.5 * direction
        for layer in layers:
            noise = rng.normal(0.0, 0.35, size=dim)
            act = (base + signal) * (1.0 + 0.05 * layer) + noise
            layer_mats[str(layer)].append(act.astype(np.float32).tolist())
        meta.append(
            {
                "item_id": row["item_id"],
                "feature": row["feature"],
                "split": row["split"],
                "behavioral_gt": row["behavioral_gt"],
                "hint_present": row["hint_present"],
                "index": i,
            }
        )
    return {
        "mode": "synthetic",
        "layers": layers,
        "dim": dim,
        "activations": layer_mats,
        "meta": meta,
        "seed": seed,
        "is_synthetic": True,
    }


def try_collect_model_activations(
    items: list[dict[str, Any]],
    *,
    model_name: str,
    layers: list[int],
    seed: int = 0,
    force_synthetic: bool = False,
    revision: str | None = None,
    use_chat_template: bool = True,
) -> dict[str, Any]:
    """Attempt real capture; fall back to synthetic with an actionable message.

    ``force_synthetic`` is reserved for smoke/plumbing. Pilot defaults try a
    measured load with chat templates and a pinned revision when available.
    """
    if force_synthetic:
        out = synthetic_activations(items, layers=layers, seed=seed)
        out["fallback_reason"] = "force_synthetic=True"
        return out

    runtime = try_load_causal_lm(
        model_name, revision=revision, force_synthetic=False
    )
    if runtime is None:
        # Measured profiles must not silently finish with planted activations.
        # Callers that want smoke plumbing must pass force_synthetic=True.
        raise RuntimeError(
            f"Could not load weights for {model_name!r} revision={revision!r}. "
            "Measured collect refused synthetic substitution. "
            "Download the model, or set force_synthetic=true for smoke only."
        )

    import torch

    model = runtime.model
    tok = runtime.tokenizer
    n_layers = int(
        getattr(model.config, "num_hidden_layers", getattr(model.config, "n_layer", 0))
    )
    bad = [layer for layer in layers if layer < 0 or layer >= n_layers]
    if bad:
        raise ValueError(f"invalid layers {bad}; valid range is [0, {n_layers - 1}]")

    captured: dict[str, list[list[float]]] = {str(layer): [] for layer in layers}
    handles = []

    def make_hook(key: str):
        def _hook(_module, _inp, out):  # type: ignore[no-untyped-def]
            tensor = out[0] if isinstance(out, tuple) else out
            vec = tensor[0, -1].detach().float().cpu().numpy()
            captured[key].append(vec.tolist())

        return _hook

    blocks = model.model.layers if hasattr(model, "model") else model.transformer.h
    for layer in layers:
        handles.append(blocks[layer].register_forward_hook(make_hook(str(layer))))

    meta: list[dict[str, Any]] = []
    chat_path = "raw"
    with torch.no_grad():
        for i, row in enumerate(items):
            prompt = row["prompt"]
            if use_chat_template:
                prompt = format_chat(
                    tok,
                    prompt,
                    system="Answer the multiple-choice item carefully.",
                )
                chat_path = "chat_template" if getattr(tok, "chat_template", None) else "chat_template_unavailable"
            enc = tok(prompt, return_tensors="pt")
            enc = {k: v.to(runtime.device) for k, v in enc.items()}
            model(**enc)
            meta.append(
                {
                    "item_id": row["item_id"],
                    "feature": row["feature"],
                    "split": row["split"],
                    "behavioral_gt": row["behavioral_gt"],
                    "hint_present": row["hint_present"],
                    "index": i,
                }
            )
    for handle in handles:
        handle.remove()
    return {
        "mode": "model",
        "model_name": model_name,
        "revision": runtime.revision,
        "layers": layers,
        "dim": len(next(iter(captured.values()))[0]) if items else 0,
        "activations": captured,
        "meta": meta,
        "seed": seed,
        "is_synthetic": False,
        "chat_template_path": chat_path,
        "notes": list(runtime.notes),
    }
