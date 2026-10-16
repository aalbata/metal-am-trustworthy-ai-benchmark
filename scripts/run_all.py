"""Run the complete benchmark pipeline from the raw Zenodo archives.

Steps (in order): prepare, splits, train, predictive, calibrate, uncertainty,
robustness, stats, baselines, xai, efficiency, tables, figures.  Every training
run is started in a fresh Python process with ``CUBLAS_WORKSPACE_CONFIG=:4096:8``;
completed runs (with test_metrics.json) are skipped.  Results are written to
``<output root>/results`` and tables and figures are generated from there.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import subprocess

import _setup
from amtrust import splits, train

STEPS = ["prepare", "splits", "train", "predictive", "calibrate", "uncertainty", "robustness", "stats",
         "baselines", "xai", "efficiency", "tables", "figures"]
SCRIPTS = Path(__file__).resolve().parent


def call(script: str, common: list[str], extra: list[str] | None = None) -> None:
    command = [sys.executable, "-I", str(SCRIPTS / script), *common, *(extra or [])]
    print("$ " + " ".join(command))
    completed = subprocess.run(command, env=train.child_environment(), check=False)
    if completed.returncode != 0:
        raise SystemExit(f"Step failed: {script} (exit code {completed.returncode})")


def main() -> int:
    p = _setup.parser(__doc__.splitlines()[0])
    p.add_argument("--steps", default=",".join(STEPS), help="Comma-separated subset of: " + ", ".join(STEPS))
    p.add_argument("--weights-dir", default=None)
    args = p.parse_args()
    layout, config = _setup.context(args)
    common = ["--data-root", str(layout.data), "--output-root", str(layout.outputs)]
    if args.config:
        common += ["--config", args.config]
    weights = ["--weights-dir", args.weights_dir] if args.weights_dir else []
    steps = [s.strip() for s in args.steps.split(",") if s.strip()]
    unknown = set(steps) - set(STEPS)
    if unknown:
        print(f"Unknown steps: {sorted(unknown)}")
        return 1
    if "prepare" in steps:
        call("prepare_data.py", common)
    if "splits" in steps:
        call("make_splits.py", common)
    master = splits.load_master(layout.master_split_csv)
    runs = train.run_manifest(master, config)
    if "train" in steps:
        for run in runs:
            if (layout.run_dir(run["run_id"]) / "test_metrics.json").is_file():
                print(f"[{run['manifest_index']:02d}] already complete: {run['run_id']}")
                continue
            call("train.py", common, ["--index", str(run["manifest_index"]), *weights])
    if "predictive" in steps:
        records = train.load_run_records(layout.runs_dir, runs)
        layout.results_dir.mkdir(parents=True, exist_ok=True)
        train.predictive_groups(records, int(config["metrics"]["group_sd_ddof"])).to_csv(
            layout.results_dir / "predictive_groups.csv", index=False, encoding="utf-8")
        print(f"predictive_groups.csv written from {len(records)} runs")
    for step, script, extra in (("calibrate", "calibrate.py", []), ("uncertainty", "uncertainty.py", []),
                                ("robustness", "robustness.py", []), ("stats", "stats.py", []),
                                ("baselines", "baselines.py", weights), ("xai", "xai.py", []),
                                ("efficiency", "efficiency.py", [])):
        if step in steps:
            call(script, common, extra)
    results = ["--results-dir", str(layout.results_dir)]
    if "tables" in steps:
        call("make_tables.py", common, results)
    if "figures" in steps:
        call("make_figures.py", common, results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
