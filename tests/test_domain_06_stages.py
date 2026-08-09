from types import SimpleNamespace
from introspect.stages import STAGES
import json

def cfg(tmp):
    return SimpleNamespace(run=SimpleNamespace(seed=0, profile="smoke"), data=SimpleNamespace(n_items=45, name="synthetic"), model=SimpleNamespace(name="no/model", require_weights=False), eval=SimpleNamespace(layers=[1,2]), experiment=SimpleNamespace(holdout_feature="sycophantic_agreement"), paths=SimpleNamespace(results=str(tmp/"results")))

def test_registry():
    assert set(STAGES) == {"build_dataset", "collect", "fit", "evaluate", "report"}

def test_e2e(tmp_path):
    c = cfg(tmp_path); run = tmp_path/"run"
    for name in ["build_dataset", "collect", "fit", "evaluate", "report"]:
        out = STAGES[name](c, run)
        assert out["task"] == name and "metrics" in out and "git_sha" in out

def test_artifacts(tmp_path):
    c = cfg(tmp_path); run = tmp_path/"run"
    for name in ["build_dataset", "collect", "fit", "evaluate"]:
        STAGES[name](c, run)
    assert (run/"artifacts/fit/probes.json").is_file()
    ev = json.loads((run/"artifacts/evaluate/results.json").read_text())
    assert ev["metrics"]["probe_agreement_is_not_evidence"] is True
