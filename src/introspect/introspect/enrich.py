"""Feature-tagged enrichment for introspection items."""
from __future__ import annotations
from typing import Any
FEATURES = ["hint_reliance","answer_certainty","format_compliance","planted_topic","sycophantic_agreement"]

def enrich_items(items: list[dict[str, Any]], cfg) -> list[dict[str, Any]]:
    out = []
    for i, row in enumerate(items):
        feat = FEATURES[i % len(FEATURES)]
        r = dict(row)
        r["feature"] = feat
        r["behavioral_gt"] = int(row.get("label", 0))
        r["components"] = {**row.get("components", {}), "feature_id": float(FEATURES.index(feat))}
        out.append(r)
    return out

