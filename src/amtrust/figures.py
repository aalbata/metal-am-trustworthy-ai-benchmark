"""Journal figures generated from the results CSV files.

Every figure is written as a vector PDF and a 600 dpi PNG preview.  Widths are
129 mm (full text width) or 84 mm (single column); text uses Arial with DejaVu
Sans as fallback at 8 pt (7 pt minimum), lines are at least 0.5 pt, and no
figure carries a title.  Architecture colours come from a colour-blind-safe
categorical set and are paired with distinct marker shapes; heatmaps use the
perceptually uniform ``cividis`` and ``viridis`` maps.  PDF metadata name only
Matplotlib as creator and producer.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from . import results_io as rio  # noqa: E402
from .tables import efficiency_columns, first_value  # noqa: E402

MM = 1.0 / 25.4
DOUBLE = 129 * MM
SINGLE = 84 * MM
ARCH_COLOURS = {"resnet18": "#0072B2", "resnet50": "#D55E00", "efficientnet_b0": "#009E73",
                "mobilenet_v3_large": "#882255"}
ARCH_MARKERS = {"resnet18": "o", "resnet50": "s", "efficientnet_b0": "D", "mobilenet_v3_large": "^"}
PDF_METADATA = {"Creator": "Matplotlib", "Producer": "Matplotlib", "Author": None, "Title": None,
                "Subject": None, "Keywords": None, "CreationDate": None}
INK = "#222222"
MUTED = "#6b6b6b"


def apply_style(font_size: float = 8.0) -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": font_size,
        "axes.labelsize": font_size,
        "axes.titlesize": font_size,
        "xtick.labelsize": font_size,
        "ytick.labelsize": font_size,
        "legend.fontsize": font_size,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "lines.linewidth": 0.8,
        "grid.linewidth": 0.5,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "text.color": INK,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "figure.dpi": 100,
    })


def save(fig, out_dir: Path, stem: str, dpi: int = 600) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf = out_dir / f"{stem}.pdf"
    png = out_dir / f"{stem}.png"
    fig.savefig(pdf, format="pdf", metadata=PDF_METADATA)
    fig.savefig(png, format="png", dpi=dpi)
    plt.close(fig)
    return [pdf, png]


def _text_colour(rgba) -> str:
    r, g, b = rgba[:3]
    return "black" if (0.2126 * r + 0.7152 * g + 0.0722 * b) > 0.45 else "white"


def panel_label(ax, text: str) -> None:
    ax.text(-0.02, 1.02, text, transform=ax.transAxes, ha="right", va="bottom", fontsize=8)


# ---------------------------------------------------------------------------
# (a) Macro-F1 heatmap
# ---------------------------------------------------------------------------
def macro_f1_heatmap(df: pd.DataFrame, out_dir: Path, config) -> list[Path]:
    df = rio.add_labels(df, config)
    grid = np.full((len(rio.ARCH_ORDER), len(rio.TASK_ORDER)), np.nan)
    sds = np.full_like(grid, np.nan)
    for _, r in df.iterrows():
        i, j = int(r["_arch_rank"]), int(r["_task_rank"])
        grid[i, j], sds[i, j] = float(r["macro_f1_mean"]), float(r["macro_f1_sd"])
    fig, ax = plt.subplots(figsize=(DOUBLE, 52 * MM), constrained_layout=True)
    cmap = plt.get_cmap("cividis")
    image = ax.imshow(grid, cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if np.isnan(grid[i, j]):
                continue
            colour = _text_colour(cmap(grid[i, j]))
            ax.text(j, i, f"{grid[i, j]:.4f}\n± {sds[i, j]:.4f}", ha="center", va="center",
                    fontsize=7, color=colour, linespacing=1.1)
    ax.set_xticks(range(len(rio.TASK_ORDER)), [t.replace(" ", "\n", 1) if t.startswith("R") else
                                               t.replace(" ", "\n", 1) for t in rio.TASK_ORDER])
    ax.set_yticks(range(len(rio.ARCH_ORDER)), [rio.ARCH_LABELS[a] for a in rio.ARCH_ORDER])
    ax.set_xlabel("Protocol, modality and held-out build")
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(-0.5, grid.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, grid.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.0)
    ax.tick_params(which="minor", length=0)
    ax.axvline(1.5, color="white", linewidth=2.5)
    bar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.01)
    bar.set_label("Mean Macro-F1 (three seeds)")
    bar.outline.set_linewidth(0.5)
    return save(fig, out_dir, "fig_macro_f1_heatmap")


# ---------------------------------------------------------------------------
# (b) Robustness direction heatmap
# ---------------------------------------------------------------------------
def robustness_direction_heatmap(df: pd.DataFrame, out_dir: Path) -> list[Path]:
    df = df.copy()
    df["_family_rank"] = df["family"].map({f: i for i, f in enumerate(rio.FAMILY_ORDER)})
    df = df.sort_values(["_family_rank", "severity"]).reset_index(drop=True)
    metrics = [("macro_f1_dec", "Macro-F1"), ("balanced_accuracy_dec", "Balanced\naccuracy"), ("auroc_dec", "AUROC")]
    grid = df[[m for m, _ in metrics]].to_numpy(dtype=float)
    total = int(df["groups"].max()) if "groups" in df.columns else 32
    labels = [f"{rio.FAMILY_LABELS.get(f, f)}, sev. {str(s).replace('S', '')}" for f, s in zip(df["family"], df["severity"])]
    fig, ax = plt.subplots(figsize=(SINGLE, 110 * MM), constrained_layout=True)
    cmap = plt.get_cmap("viridis")
    image = ax.imshow(grid, cmap=cmap, vmin=0, vmax=total, aspect="auto")
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            ax.text(j, i, f"{int(grid[i, j])}", ha="center", va="center", fontsize=7,
                    color=_text_colour(cmap(grid[i, j] / total)))
    ax.set_xticks(range(len(metrics)), [label for _, label in metrics])
    ax.set_yticks(range(len(labels)), labels)
    ax.xaxis.tick_top()
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(-0.5, grid.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, grid.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.0)
    ax.tick_params(which="minor", length=0)
    for boundary in (2.5, 5.5, 8.5, 11.5):
        ax.axhline(boundary, color="white", linewidth=2.5)
    bar = fig.colorbar(image, ax=ax, orientation="horizontal", fraction=0.04, pad=0.02)
    bar.set_label(f"Groups (of {total}) with lower\ncorrupted performance")
    bar.set_ticks([0, 8, 16, 24, total])
    bar.outline.set_linewidth(0.5)
    return save(fig, out_dir, "fig_robustness_direction")


# ---------------------------------------------------------------------------
# (c) Efficiency scatter
# ---------------------------------------------------------------------------
def efficiency_scatter(df: pd.DataFrame, out_dir: Path) -> list[Path]:
    cols = efficiency_columns(df)
    fig, ax = plt.subplots(figsize=(SINGLE, 78 * MM), layout="constrained")
    precision_markers = {"FP32": "o", "AMP/FP16": "^"}
    for _, r in df.iterrows():
        arch = rio.architecture_key(r["architecture"])
        prec = rio.precision_label(r["precision"])
        ax.scatter(first_value(r, cols["b1_mean"]), first_value(r, cols["b8_throughput"]), s=30,
                   marker=precision_markers.get(prec, "o"), color=ARCH_COLOURS.get(arch, MUTED),
                   edgecolor="white", linewidth=0.6, zorder=3)
    ax.set_xlabel("Batch-1 mean latency (ms per image)")
    ax.set_ylabel("Batch-8 throughput (images/s)")
    ax.grid(True, color="#e3e3e3", linewidth=0.5, zorder=0)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    arch_handles = [Line2D([], [], linestyle="none", marker="s", markersize=5, color=ARCH_COLOURS[a],
                           label=rio.ARCH_LABELS[a]) for a in rio.ARCH_ORDER]
    prec_handles = [Line2D([], [], linestyle="none", marker=m, markersize=5, color=MUTED, label=p)
                    for p, m in precision_markers.items()]
    fig.legend(handles=arch_handles + prec_handles, loc="outside lower center", ncol=3, fontsize=7,
               handletextpad=0.3, columnspacing=0.9, labelspacing=0.3)
    return save(fig, out_dir, "fig_efficiency")


# ---------------------------------------------------------------------------
# (d), (e) Dot plots per group
# ---------------------------------------------------------------------------
def _dot_panel(ax, df: pd.DataFrame, column: str, reference: float):
    offsets = {a: (k - 1.5) * 0.18 for k, a in enumerate(rio.ARCH_ORDER)}
    for _, r in df.iterrows():
        value = float(r[column])
        if np.isnan(value):
            continue
        y = int(r["_task_rank"]) + offsets[r["architecture"]]
        ax.scatter(value, y, s=16, marker=ARCH_MARKERS[r["architecture"]], color=ARCH_COLOURS[r["architecture"]],
                   edgecolor="white", linewidth=0.4, zorder=3)
    ax.axvline(reference, color=MUTED, linewidth=0.6, linestyle=(0, (3, 2)), zorder=1)
    ax.set_yticks(range(len(rio.TASK_ORDER)), rio.TASK_ORDER)
    ax.set_ylim(len(rio.TASK_ORDER) - 0.5, -0.5)
    for boundary in (1.5, 4.5):
        ax.axhline(boundary, color="#d0d0d0", linewidth=0.5, zorder=0)
    ax.grid(True, axis="x", color="#ececec", linewidth=0.5, zorder=0)


def _arch_legend(fig):
    handles = [Line2D([], [], linestyle="none", marker=ARCH_MARKERS[a], markersize=5, color=ARCH_COLOURS[a],
                      label=rio.ARCH_LABELS[a]) for a in rio.ARCH_ORDER]
    fig.legend(handles=handles, loc="outside lower center", ncol=4, fontsize=7, handletextpad=0.3, columnspacing=1.0)


def calibration_dotplot(df: pd.DataFrame, out_dir: Path, config) -> list[Path]:
    df = rio.add_labels(df, config)
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 72 * MM), sharey=True, layout="constrained")
    _dot_panel(axes[0], df, "nll_delta_mean", 0.0)
    axes[0].set_xscale("symlog", linthresh=0.01, linscale=0.6)
    values = df["nll_delta_mean"].astype(float).dropna()
    axes[0].set_xlim(min(values.min() * 1.8, -0.02), max(values.max() * 1.8, 0.02))
    axes[0].set_xticks([-0.1, 0.0, 0.1, 1.0, 10.0], ["−0.1", "0", "0.1", "1", "10"])
    axes[0].set_xlabel("Change in NLL (nats,\nsymmetric logarithmic axis)")
    _dot_panel(axes[1], df, "brier_delta_mean", 0.0)
    axes[1].set_xlabel("Change in Brier score\n")
    axes[1].tick_params(labelleft=False)
    panel_label(axes[0], "(a)")
    panel_label(axes[1], "(b)")
    _arch_legend(fig)
    return save(fig, out_dir, "fig_calibration_change")


def uncertainty_dotplot(df: pd.DataFrame, out_dir: Path, config) -> list[Path]:
    df = rio.add_labels(df, config)
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 72 * MM), sharey=True, sharex=True, layout="constrained")
    _dot_panel(axes[0], df, "error_auroc_entropy", 0.5)
    axes[0].set_xlabel("Error-detection AUROC,\npredictive entropy")
    _dot_panel(axes[1], df, "error_auroc_entropy_gap", 0.5)
    axes[1].set_xlabel("Error-detection AUROC,\nthree-seed entropy gap")
    axes[1].tick_params(labelleft=False)
    axes[0].set_xlim(0.3, 1.0)
    panel_label(axes[0], "(a)")
    panel_label(axes[1], "(b)")
    _arch_legend(fig)
    return save(fig, out_dir, "fig_uncertainty_auroc")


def make_all(results_dir: Path, out_dir: Path, config) -> dict:
    """Create every figure whose source CSV exists; return {figure stem: status}."""
    apply_style(float(config["figures"]["font_size_pt"]))
    status = {}
    jobs = [
        ("fig_macro_f1_heatmap", "predictive_groups.csv", lambda f: macro_f1_heatmap(f, out_dir, config)),
        ("fig_robustness_direction", "robustness_direction.csv", lambda f: robustness_direction_heatmap(f, out_dir)),
        ("fig_efficiency", "efficiency.csv", lambda f: efficiency_scatter(f, out_dir)),
        ("fig_calibration_change", "calibration_groups.csv", lambda f: calibration_dotplot(f, out_dir, config)),
        ("fig_uncertainty_auroc", "uq_groups.csv", lambda f: uncertainty_dotplot(f, out_dir, config)),
    ]
    for stem, source, job in jobs:
        frame = rio.load(source, results_dir)
        if frame is None:
            status[stem] = f"skipped (results/{source} not found)"
            continue
        job(frame)
        status[stem] = "written"
    return status
