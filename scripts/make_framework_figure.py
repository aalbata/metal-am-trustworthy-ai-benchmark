"""Draw the evaluation framework figure (survey, benchmark, implications)."""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 7,
    "pdf.fonttype": 42,
})

MM = 1 / 25.4
W, H = 129, 112
fig = plt.figure(figsize=(W * MM, H * MM))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")

BLUE, BLUE_L = "#1f5f8b", "#e3eef6"
GREEN, GREEN_L = "#2e7d4f", "#e5f2ea"
ORANGE, ORANGE_L = "#b5651d", "#f8ecdf"
GREY = "#404040"


def box(x, y, w, h, text, edge, face, size=7, weight="normal"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.0",
                                linewidth=0.6, edgecolor=edge, facecolor=face))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, fontweight=weight,
            linespacing=1.2)


def arrow(x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=7,
                                 linewidth=0.7, color=GREY))


def band(y, h, title, edge, face):
    ax.add_patch(FancyBboxPatch((1, y), W - 2, h, boxstyle="round,pad=0,rounding_size=1.5",
                                linewidth=0.6, edgecolor=edge, facecolor=face))
    ax.text(3, y + h - 3.2, title, ha="left", va="center", fontsize=7.5)


def chips(items, x0, y, w, h, edge, cols, gap=1.6, size=6.8):
    cw = (w - gap * (cols - 1)) / cols
    for i, t in enumerate(items):
        r, c = divmod(i, cols)
        box(x0 + c * (cw + gap), y - r * (h + gap), cw, h, t, edge, "white", size)


# survey band
band(76, 34, "Structured critical survey (RQ1, RQ2)", BLUE, BLUE_L)
chips(["Generalisation and\nevaluation validity", "Calibration and\nuncertainty", "Robustness",
       "Explainability", "Reproducibility", "Computational\nefficiency"], 3, 94.5, 123, 8, BLUE, 3)
ax.text(W / 2, 79.5, "Search, screening, stratified random sample, full-text coding of evaluation practice",
        ha="center", va="center", fontsize=6.8)
arrow(W / 2, 75.6, W / 2, 72.4)

# benchmark band
band(38, 34, "Controlled benchmark (RQ3)", GREEN, GREEN_L)
chips(["Public LPBF dataset, three builds,\nOT and PB images", "Four CNNs, three seeds,\n96 runs",
       "Protocol R (random split) versus\nProtocol S (held-out build)"], 3, 56.5, 123, 9, GREEN, 3)
chips(["Predictive\nperformance", "Calibration", "Uncertainty", "Robustness", "Grad-CAM\ndescriptors",
       "Inference\ncost"], 3, 43.5, 123, 8.5, GREEN, 6, size=6.5)
arrow(W / 2, 37.6, W / 2, 34.4)

# implications band
band(4, 30, "Implications for manufacturing evaluation", ORANGE, ORANGE_L)
chips(["Hold out the unit over which\nthe model will be used", "Report ranking and threshold\nmetrics; plan thresholds",
       "Re-establish calibration in\neach new context", "Tie robustness and explanation\ntests to failure modes",
       "Measure timing end to end\non target hardware", "Evidence ladder from sampled\ndata to closed-loop use"],
      3, 18.0, 123, 9, ORANGE, 3, size=6.5)

meta = {"Creator": "Matplotlib", "Producer": "Matplotlib", "Author": None, "Title": None,
        "Subject": None, "Keywords": None, "CreationDate": None}
fig.savefig(OUT / "Fig1.pdf", metadata=meta)
fig.savefig(OUT / "Fig1.png", dpi=600)
