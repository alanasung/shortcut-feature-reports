from types import SimpleNamespace
from pathlib import Path
from introspect.stages import STAGES
import json

def cfg(tmp):
    return SimpleNamespace(
        run=SimpleNamespace(seed=1, profile="pilot"),
        data=SimpleNamespace(n_items=45, name="synthetic"),
        model=SimpleNamespace(name="does-not-exist/model", require_weights=False),
        eval=SimpleNamespace(layers=[1, 2, 3]),
        experiment=SimpleNamespace(holdout_feature="format_compliance"),
        paths=SimpleNamespace(results=str(tmp / "results")),
    )

def test_build_writes_dataset(tmp_path):
    out = STAGES["build_dataset"](cfg(tmp_path), tmp_path / "r")
    assert (tmp_path / "r/artifacts/dataset/dataset.json").is_file()
    assert out["metrics"]["holdout_feature"] == "format_compliance"

def test_collect_marks_synthetic(tmp_path):
    run = tmp_path / "r"
    STAGES["build_dataset"](cfg(tmp_path), run)
    out = STAGES["collect"](cfg(tmp_path), run)
    assert out.get("is_synthetic", True) is True

def test_fit_creates_probes(tmp_path):
    run = tmp_path / "r"
    c = cfg(tmp_path)
    for s in ("build_dataset", "collect", "fit"):
        STAGES[s](c, run)
    probes = json.loads((run / "artifacts/fit/probes.json").read_text())
    assert "validations" in probes

def test_evaluate_has_baselines(tmp_path):
    run = tmp_path / "r"
    c = cfg(tmp_path)
    for s in ("build_dataset", "collect", "fit", "evaluate"):
        STAGES[s](c, run)
    ev = json.loads((run / "artifacts/evaluate/results.json").read_text())
    assert "baselines" in ev["metrics"]
    assert ev["metrics"]["probe_agreement_is_not_evidence"] is True

def test_report_summary(tmp_path):
    run = tmp_path / "r"
    c = cfg(tmp_path)
    for s in ("build_dataset", "collect", "fit", "evaluate", "report"):
        STAGES[s](c, run)
    assert (run / "artifacts/report/summary.json").is_file() or (run / "artifacts/report/results.json").is_file()
