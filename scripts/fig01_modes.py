"""Figure 1: barotropic and baroclinic modes of the two-layer aquifer."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(
    __file__))))

from coastal_interface import linear, plotting  # noqa: E402
from coastal_interface.io_utils import write_csv  # noqa: E402
from coastal_interface.scenario import (M2_PERIOD_DAYS,  # noqa: E402
                                        REFERENCE_AQUIFER, YEAR_DAYS)

plt = plotting.plt


def main():
    """Mode diffusivities against layer split and decay lengths."""
    plotting.use_style()
    a = REFERENCE_AQUIFER
    total = 20.0
    c = a.conductivity / a.porosity

    split = np.linspace(0.005, 0.995, 199)
    d0, d1 = [], []
    for r in split:
        layers = linear.Layers.from_aquifer(a, r * total, (1 - r) * total)
        exact = linear.mode_diffusivities(layers)
        d0.append(exact[0] / (c * total))
        d1.append(exact[1] / (a.delta * c * total / 4.0))
    d0, d1 = np.array(d0), np.array(d1)

    layers = linear.Layers.from_aquifer(a, 10.0, 10.0)
    periods = np.logspace(-1.0, 4.0, 300)
    lengths = np.array([linear.skin_depths(layers, 2 * np.pi / p)
                        for p in periods])

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(plotting.DOUBLE, 2.5),
                                 gridspec_kw={"wspace": 0.32})
    ax.plot(split, d0, color=plotting.FLUX, label=r"$D_0 / (KH/n)$")
    ax.plot(split, d1, color=plotting.HEAD,
            label=r"$D_1 / (\delta K H / 4n)$")
    ax.plot(split, 4 * split * (1 - split), color=plotting.INK, lw=0.8,
            ls="--", label=r"$4 H_f H_s / H^2$ (small $\delta$)")
    ax.set_xlabel(r"freshwater fraction of the saturated thickness, $H_f/H$")
    ax.set_ylabel("normalized diffusivity")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.12)
    plotting.legend_below(ax)
    ax.text(0.5, 1.045, r"$D_0 \approx KH/n$", ha="center",
            color=plotting.FLUX, fontsize=7)
    plotting.panel_label(ax, "a")

    bx.loglog(periods, lengths[:, 0], color=plotting.FLUX,
              label=r"barotropic, $L_0$")
    bx.loglog(periods, lengths[:, 1], color=plotting.HEAD,
              label=r"baroclinic (interface), $L_1$")
    bx.axhline(total, color=plotting.INK, lw=0.8, ls="--",
               label="saturated thickness H")
    marks = [(M2_PERIOD_DAYS, "M2"), (14.77, "spring-neap"),
             (YEAR_DAYS, "annual")]
    for period, name in marks:
        bx.axvline(period, color=plotting.MUTED, lw=0.6, ls=":")
        bx.text(period * 1.12, 0.12, name, rotation=90, fontsize=6.5,
                color=plotting.MUTED, va="bottom")
    bx.set_xlabel("forcing period (days)")
    bx.set_ylabel(r"decay length $\sqrt{2D_j/\omega}$ (m)")
    bx.set_ylim(0.1, 3e4)
    plotting.legend_below(bx)
    plotting.panel_label(bx, "b")

    plotting.save(fig, "fig01_modes")
    write_csv("fig01a_diffusivities", {
        "fresh_fraction": split, "D0_over_KH_n": d0,
        "D1_over_deltaKH_4n": d1},
        "exact mode diffusivities, delta = 0.025")
    write_csv("fig01b_decay_lengths", {
        "period_days": periods, "L0_m": lengths[:, 0], "L1_m": lengths[:, 1]},
        "H_f = H_s = 10 m, K = 30 m/day, n = 0.3")


if __name__ == "__main__":
    main()
