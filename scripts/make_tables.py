"""Generate LaTeX (booktabs) table snippets and matching CSV tables from the results CSV files.

Default input is the repository ``results`` folder and default output is
``<output root>/tables``.  Tables whose source CSV is absent are skipped.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _setup
from amtrust import tables


def main() -> int:
    p = _setup.parser(__doc__.splitlines()[0])
    p.add_argument("--results-dir", default=None, help="Folder with the results CSV files (default: <repo>/results).")
    p.add_argument("--out-dir", default=None, help="Output folder (default: <output root>/tables).")
    args = p.parse_args()
    layout, config = _setup.context(args)
    results_dir = Path(args.results_dir) if args.results_dir else _setup.RESULTS_DIR
    out_dir = Path(args.out_dir) if args.out_dir else layout.tables_dir
    status = tables.make_all(results_dir, out_dir, config)
    for stem, state in status.items():
        print(f"{stem:<28} {state}")
    print(f"Tables folder: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
