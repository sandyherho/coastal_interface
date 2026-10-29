"""Shared figure style and saving."""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.transforms import ScaledTranslation  # noqa: E402

from .io_utils import FIGURES, ensure_dirs  # noqa: E402

# Categorical colors in fixed order: flux-controlled, head-controlled,
# third series. Theory is drawn in ink, never in a series color.
FLUX = "#2a78d6"
HEAD = "#eb6834"
THIRD = "#1baf7a"
INK = "#0b0b0b"
MUTED = "#6f6e69"
GRID = "#e4e3df"

# Fills of the cross-sections: fresh water, salt water, aquifer base.
FRESH_FILL = "#d6e8f8"
SALT_FILL = "#8fb3cf"
BASE_FILL = "#c9c4b8"
SEA_FILL = "#5f8fb4"

SINGLE = 3.4
DOUBLE = 7.0


def use_style():
    """Apply the figure style used throughout."""
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.linewidth": 0.6,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "lines.linewidth": 1.4,
        "lines.markersize": 4,
        "legend.frameon": False,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
        "figure.dpi": 150,
    })


def panel_label(ax, letter):
    """Bold panel letter in the upper left corner."""
    ax.text(-0.02, 1.02, f"({letter})", transform=ax.transAxes,
            ha="right", va="bottom", fontweight="bold", color=INK)


def legend_below(ax, ncol=1, offset=30.0, **kwargs):
    """Legend centered under a panel, outside it, below the x label.

    ``offset`` is the distance in points from the bottom of the axes to
    the top of the legend; the default clears the tick labels and the
    axis label at the font sizes used here.
    """
    shift = ScaledTranslation(0.0, -offset / 72.0,
                              ax.figure.dpi_scale_trans)
    return ax.legend(loc="upper center", bbox_to_anchor=(0.5, 0.0),
                     bbox_transform=ax.transAxes + shift, ncol=ncol,
                     borderaxespad=0.0, columnspacing=1.2,
                     handlelength=1.8, **kwargs)


def save(fig, name):
    """Save as vector PDF and 600 dpi PNG in outputs/figures."""
    ensure_dirs()
    fig.savefig(os.path.join(FIGURES, f"{name}.pdf"))
    fig.savefig(os.path.join(FIGURES, f"{name}.png"), dpi=600)
    plt.close(fig)


def draw_lens(ax, x, h, zeta, base_depth, sea_level=0.0, x_max=None):
    """Shade fresh water, salt water and the base of a cross-section."""
    x_max = x[-1] if x_max is None else x_max
    ax.fill_between(x, zeta, h, color=FRESH_FILL, lw=0)
    ax.fill_between(x, -base_depth, zeta, color=SALT_FILL, lw=0)
    ax.axhspan(-base_depth - 2.0, -base_depth, color=BASE_FILL, lw=0)
    ax.fill_between([-0.06 * x_max, 0.0], [-base_depth] * 2,
                    [sea_level] * 2, color=SEA_FILL, lw=0)
    ax.plot(x, h, color=INK, lw=0.9)
    ax.plot(x, zeta, color=INK, lw=0.9)
