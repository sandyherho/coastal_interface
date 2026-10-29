"""Animation 2: four years of monsoon-swell setup on both aquifers."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(
    __file__))))

from coastal_interface import anim, experiments as ex  # noqa: E402
from coastal_interface import plotting  # noqa: E402
from coastal_interface.scenario import REFERENCE_AQUIFER  # noqa: E402

plt = plotting.plt
X_MAX = 140.0


def main():
    """Tidal-mean lenses of the flux- and head-controlled aquifers."""
    plotting.use_style()
    a = REFERENCE_AQUIFER
    runs = {k: ex.seasonal(kind=k, tide_amplitude=0.5)
            for k in ("flux", "head")}
    frames = np.arange(0, len(runs["head"]["frame_t"]), 2)
    x_full = runs["head"]["x"]
    view = x_full < X_MAX
    x = np.concatenate(([0.0], x_full[view]))

    fig = plt.figure(figsize=(7.0, 3.6))
    lens_axes = [fig.add_axes([0.08, 0.56, 0.55, 0.36]),
                 fig.add_axes([0.08, 0.11, 0.55, 0.36])]
    series = fig.add_axes([0.72, 0.11, 0.26, 0.81])
    colors = {"flux": plotting.FLUX, "head": plotting.HEAD}
    for kind in ("flux", "head"):
        run = runs[kind]
        series.plot(run["t"] / 365.25, run["toe"], color=colors[kind],
                    lw=1.0, label=f"{kind}-controlled")
        series.plot(run["t"] / 365.25, run["toe_quasi_static"],
                    color=colors[kind], lw=0.6, ls="--")
    series.set_xlabel("time (years)")
    series.set_ylabel("toe position (m)")
    series.legend(loc="upper left", fontsize=6.5)
    series.text(0.98, 0.02, "dashed: quasi-static", transform=series.transAxes,
                ha="right", fontsize=6.5, color=plotting.MUTED)
    cursor = series.axvline(0.0, color=plotting.INK, lw=0.6)

    def update(i):
        k = frames[i]
        t = runs["head"]["frame_t"][k]
        setup = float(np.interp(t, runs["head"]["t"], runs["head"]["setup"]))
        for axis, kind in zip(lens_axes, ("flux", "head")):
            run = runs[kind]
            axis.clear()
            h = np.concatenate(([setup], run["frame_h"][k][view]))
            z = np.concatenate(([setup], run["frame_z"][k][view]))
            plotting.draw_lens(axis, x, h, z, a.base_depth,
                               sea_level=setup, x_max=X_MAX)
            toe = float(np.interp(t, run["t"], run["toe"]))
            axis.axvline(toe, color=colors[kind], lw=1.2)
            axis.axvline(float(run["toe0"]), color=plotting.INK, lw=0.6,
                         ls=":")
            axis.set_xlim(-0.06 * X_MAX, X_MAX)
            axis.set_ylim(-a.base_depth - 1.0, 2.0)
            axis.set_ylabel("elevation (m)")
            axis.grid(False)
            axis.text(0.99, 0.55, f"{kind}-controlled, toe {toe:5.1f} m",
                      transform=axis.transAxes, ha="right", fontsize=7,
                      color=plotting.INK)
        lens_axes[1].set_xlabel("distance from shoreline (m)")
        lens_axes[0].set_title(f"year {t / 365.25:4.2f}    tidal-mean "
                               f"setup {100 * setup:4.1f} cm",
                               fontsize=8, loc="left")
        cursor.set_xdata([t / 365.25, t / 365.25])
        return []

    anim.save_gif(fig, update, len(frames), "anim02_season", fps=15)


if __name__ == "__main__":
    main()
