"""
Shared publication figure style (Nature-Communications taste).

Conventions mirrored from the aon_pir_rev paper repo:
  - Arial sans-serif, modest font sizes
  - top / right spines hidden, ticks pointing outward
  - no gridlines
  - editable vector text (pdf.fonttype = 42) so figures can be tweaked in
    Illustrator/Inkscape at submission
  - figures saved as both vector PDF and high-DPI PNG
"""

import matplotlib as mpl


def apply_style():
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "axes.labelsize": 9,
        "axes.linewidth": 1.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 4, "ytick.major.size": 4,
        "xtick.major.width": 1.0, "ytick.major.width": 1.0,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def clean_spines(ax):
    """Hide top/right spines and set outward ticks on a single axis."""
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.tick_params(direction="out", length=4, width=1.0)


def panel_label(ax, letter, dx=-0.08, dy=1.02, fontsize=13):
    """Bold panel identifier at the top-left corner (outside the axes), matching Fig 1.

    Use this instead of embedding '(A)' in the panel title, so every figure carries a
    consistent, professional panel-letter scheme (one distinct letter per panel).
    """
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=fontsize,
            fontweight="bold", va="bottom", ha="right")


def save_fig(fig, stem, formats=("pdf", "png")):
    """Save a figure to <stem>.<ext> for each requested format."""
    for ext in formats:
        fig.savefig(f"{stem}.{ext}", bbox_inches="tight")
