"""Generate the data figures (vector PDF and 600 dpi PNG) from the results CSV files.

Figures: Macro-F1 heatmap (predictive_groups.csv), robustness direction
heatmap (robustness_direction.csv), efficiency scatter (efficiency.csv),
calibration change dot plot (calibration_groups.csv) and uncertainty
error-detection AUROC dot plot (uq_groups.csv).  Figures whose source CSV is
absent are skipped.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _setup
from amtrust import figures


def main() -> int:
    p = _setup.parser(__doc__.splitlines()[0])
    p.add_argument("--results-dir", default=None, help="Folder with the results CSV files (default: <repo>/results).")
    p.add_argument("--out-dir", default=None, help="Output folder (default: <output root>/figures).")
    args = p.parse_args()
    layout, config = _setup.context(args)
    results_dir = Path(args.results_dir) if args.results_dir else _setup.RESULTS_DIR
    out_dir = Path(args.out_dir) if args.out_dir else layout.figures_dir
    status = figures.make_all(results_dir, out_dir, config)
    for stem, state in status.items():
        print(f"{stem:<28} {state}")
    print(f"Figures folder: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
