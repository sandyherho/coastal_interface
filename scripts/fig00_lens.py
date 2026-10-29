"""Figure 0: the idealized aquifer and its steady freshwater lens."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(
    __file__))))

from coastal_interface import core, experiments as ex, plotting  # noqa
from coastal_interface.io_utils import write_csv  # noqa: E402
from coastal_interface.scenario import REFERENCE_AQUIFER  # noqa: E402

plt = plotting.plt


def main():
    """Draw the reference lens with the quantities used in the paper."""
    plotting.use_style()
    a = REFERENCE_AQUIFER
    grid = ex.make_grid()
    inland = ex.make_inland("flux", a, ex.REFERENCE_DISCHARGE, grid.length)
    y = ex.initial_lens(grid, a, inland, ex.REFERENCE_DISCHARGE)
    x, h, z = grid.centers, y[0::2], y[1::2]
    toe = core.toe_position(x, h, z, a.delta, a.base_depth)
    # Add the shoreline point, where the lens closes at sea level.
    x_plot = np.concatenate(([0.0], x))
    h_plot = np.concatenate(([0.0], h))
    z_plot = np.concatenate(([0.0], z))

    fig, (ax, zoom) = plt.subplots(
        1, 2, figsize=(plotting.DOUBLE, 2.4),
        gridspec_kw={"width_ratios": [2.3, 1.0], "wspace": 0.28})

    plotting.draw_lens(ax, x_plot, h_plot, z_plot, a.base_depth,
                       x_max=150.0)
    ax.set_xlim(-9.0, 150.0)
    ax.set_ylim(-a.base_depth - 1.5, 2.5)
    ax.set_xlabel("distance from shoreline, x (m)")
    ax.set_ylabel("elevation above mean sea level (m)")
    ax.grid(False)
    ax.vlines(toe, -a.base_depth, np.interp(toe, x, h), color=plotting.MUTED,
              lw=0.6, ls=":")
    style = dict(fontsize=7, color=plotting.INK)
    ax.text(95, 1.2, "water table h(x, t)", **style)
    ax.annotate(r"interface $\zeta(x, t)$", xy=(40, np.interp(40, x, z)),
                xytext=(52, -6.0), arrowprops=dict(arrowstyle="-", lw=0.5),
                **style)
    ax.text(100, -9, "fresh water", ha="center", **style)
    ax.text(25, -17, "salt water", ha="center", **style)
    ax.text(toe + 1.5, -a.base_depth + 1.0, f"toe $x_T$ = {toe:.1f} m",
            **style)
    ax.text(-4.5, -10, r"sea  $\eta(t)$", rotation=90, ha="center",
            va="center", color="white", fontsize=7)
    ax.text(75, -a.base_depth - 1.1, "impermeable base, z = -B", ha="center",
            **style)
    ax.annotate("", xy=(130, -2.0), xytext=(146, -2.0),
                arrowprops=dict(arrowstyle="->", lw=0.6))
    ax.text(138, -1.6, "Q", ha="center", va="bottom", **style)
    plotting.panel_label(ax, "a")

    plotting.draw_lens(zoom, x_plot, h_plot, z_plot, a.base_depth,
                       x_max=6.0)
    zoom.set_xlim(-0.36, 6.0)
    zoom.set_ylim(-5.0, 0.6)
    zoom.set_xlabel("x (m)")
    zoom.grid(False)
    zoom.text(1.9, -4.6, r"$\eta - \zeta = (h - \eta)/\delta$",
              ha="center", **style)
    zoom.text(3.6, -1.2, "lens pinches out\nat the shoreline",
              ha="center", fontsize=6.5, color=plotting.INK)
    plotting.panel_label(zoom, "b")

    plotting.save(fig, "fig00_lens")
    write_csv("fig00_lens", {"x_m": x, "h_m": h, "zeta_m": z},
              "discrete steady lens, Q = 2 m^2/day, reference aquifer")


if __name__ == "__main__":
    main()
