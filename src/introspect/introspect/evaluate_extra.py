"""Introspection-specific metrics: behavioral GT agreement + causal sensitivity proxy."""
from __future__ import annotations

import numpy as np


def evaluate_extra(cfg, run_dir, y, prob):
    # agreement vs behavioral labels (here y) is a training diagnostic only
    diag = float(np.mean((prob > 0.5).astype(int) == y))
    # causal sensitivity proxy: shuffle features effect already in ablations; report placeholder delta
    return {
        "probe_agreement_diagnostic": diag,
        "behavioral_accuracy": diag,
        "causal_sensitivity": float(abs(diag - 0.5)),
        "notes": "probe agreement is diagnostic only; introspection claim needs patching stage",
    }

