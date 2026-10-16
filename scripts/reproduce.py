"""Reproduce the reported tables and figures.

--fast  regenerates the table snippets (LaTeX and CSV) and the data figures
        from the shipped results/*.csv files only.  CPU, less than 20 seconds.
--full  runs the whole pipeline from the two Zenodo archives (scripts/run_all.py):
        cohort, splits, 96 training runs, calibration, uncertainty, robustness,
        statistics, baselines, Grad-CAM and efficiency, then tables and figures.
        Requires an NVIDIA GPU with CUDA; the total runtime has not been measured
        on current hardware and is expected to be long (many GPU hours).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import subprocess
import time

import _setup
from amtrust import figures, tables


def main() -> int:
    p = _setup.parser(__doc__.splitlines()[0])
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--fast", action="store_true")
    mode.add_argument("--full", action="store_true")
    p.add_argument("--results-dir", default=None, help="Results folder for --fast (default: <repo>/results).")
    p.add_argument("--weights-dir", default=None, help="Folder with ImageNet weights files for --full.")
    args = p.parse_args()
    layout, config = _setup.context(args)
    if args.fast:
        start = time.perf_counter()
        results_dir = Path(args.results_dir) if args.results_dir else _setup.RESULTS_DIR
        table_status = tables.make_all(results_dir, layout.tables_dir, config)
        figure_status = figures.make_all(results_dir, layout.figures_dir, config)
        for stem, state in {**table_status, **figure_status}.items():
            print(f"{stem:<28} {state}")
        print(f"Tables : {layout.tables_dir}")
        print(f"Figures: {layout.figures_dir}")
        print(f"Elapsed: {time.perf_counter() - start:.1f} s")
        return 0
    command = [sys.executable, "-I", str(Path(__file__).resolve().parent / "run_all.py"),
               "--data-root", str(layout.data), "--output-root", str(layout.outputs)]
    if args.config:
        command += ["--config", args.config]
    if args.weights_dir:
        command += ["--weights-dir", args.weights_dir]
    return subprocess.run(command, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
