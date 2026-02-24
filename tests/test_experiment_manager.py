import json
from pathlib import Path

from evaluation import experiment_manager


def test_prepare_managed_run(tmp_path):
    run_dir, manifest_path = experiment_manager.prepare_managed_run(
        base_out_dir=str(tmp_path),
        benchmarks_path="evaluation/benchmarks.json",
        seeds=3,
        run_ablations=True,
        config_path="",
        argv=["run_benchmarks.py"],
        run_name="test",
    )

    manifest = json.loads(Path(manifest_path).read_text())
    assert manifest["seeds"] == 3
    assert manifest["run_ablations"] is True
    assert manifest["benchmarks"].endswith("benchmarks.json")

    latest = json.loads((tmp_path / "latest_managed_run.json").read_text())
    assert latest["run_id"] == manifest["run_id"]
