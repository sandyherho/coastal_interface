"""Figure 6: a monsoon-swell setup cycle on a tidal coast."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(
    __file__))))

from coastal_interface import experiments as ex, plotting  # noqa: E402
from coastal_interface.io_utils import write_csv  # noqa: E402

plt = plotting.plt
COLORS = {"flux": plotting.FLUX, "head": plotting.HEAD}


def main():
    """Plot the setup and the toe of both aquifers, with and without tide."""
    plotting.use_style()
    fig, axes = plt.subplots(3, 1, figsize=(plotting.DOUBLE, 4.4),
                             sharex=True,
                             gridspec_kw={"hspace": 0.18,
                                          "height_ratios": [0.7, 1, 1]})
    tided = {k: ex.seasonal(kind=k, tide_amplitude=0.5)
             for k in ("flux", "head")}
    calm = {k: ex.seasonal(kind=k, tide_amplitude=0.0)
            for k in ("flux", "head")}
    years = tided["head"]["t"] / 365.25
    axes[0].plot(years, tided["head"]["setup"], color=plotting.INK, lw=1.0)
    axes[0].set_ylabel("setup (m)")
    axes[0].set_title("tidal-mean wave setup: 0.2 m mean, 0.15 m annual "
                      "amplitude, with an M2 tide of 0.5 m on top",
                      fontsize=7, color=plotting.MUTED)
    plotting.panel_label(axes[0], "a")

    for axis, kind, letter in ((axes[1], "flux", "b"),
                               (axes[2], "head", "c")):
        run, still = tided[kind], calm[kind]
        color = COLORS[kind]
        t = run["t"] / 365.25
        axis.plot(t, run["toe_quasi_static"], color=plotting.INK, lw=0.8,
                  ls="--", label="quasi-static (Strack-Glover)")
        axis.plot(still["t"] / 365.25, still["toe"], color=plotting.MUTED,
                  lw=0.8, label="model without tide")
        axis.plot(t, run["toe"], color=color, lw=1.2,
                  label="model with tide")
        axis.axhline(float(run["toe0"]), color=plotting.MUTED, lw=0.5,
                     ls=":")
        axis.set_ylabel(r"toe $x_T$ (m)")
        axis.text(0.01, 0.9, f"{kind}-controlled",
                  transform=axis.transAxes, fontsize=7, fontweight="bold",
                  color=color, va="top")
        plotting.panel_label(axis, letter)
        write_csv(f"fig06_{kind}", {
            "t_days": run["t"], "setup_m": run["setup"],
            "toe_m": run["toe"], "toe_quasi_static_m":
            run["toe_quasi_static"],
            "toe_no_tide_m": np.interp(run["t"], still["t"], still["toe"]),
            "salt_volume_m2": run["salt_volume"]},
            f"{kind}-controlled lens, tidal-cycle means")
    handles, labels = axes[2].get_legend_handles_labels()
    fig.legend([handles[0], handles[1], handles[2]],
               [labels[0], labels[1], "model with tide"],
               loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.03))
    axes[-1].set_xlabel("time (years)")
    axes[-1].set_xlim(0, years[-1])
    plotting.save(fig, "fig06_seasonal")


if __name__ == "__main__":
    main()
