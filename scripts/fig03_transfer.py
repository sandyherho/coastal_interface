"""Figure 3: the interface as a low-pass filter of sea level."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(
    __file__))))

from coastal_interface import experiments as ex, linear, plotting  # noqa
from coastal_interface.io_utils import write_csv  # noqa: E402
from coastal_interface.scenario import (M2_PERIOD_DAYS,  # noqa: E402
                                        REFERENCE_AQUIFER, YEAR_DAYS)

plt = plotting.plt
SERIES = [plotting.FLUX, plotting.HEAD, plotting.THIRD, "#eda100"]


def main():
    """Theory of the interface transfer function and its check."""
    plotting.use_style()
    a = REFERENCE_AQUIFER
    layers = linear.Layers.from_aquifer(a, 10.0, 10.0)
    periods = np.logspace(-1.0, 4.0, 400)
    distances = (1.0, 5.0, 20.0, 80.0)

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(plotting.DOUBLE, 2.6),
                                 gridspec_kw={"wspace": 0.3})
    table = {"period_days": periods}
    for color, dist in zip(SERIES, distances):
        gain = np.array([abs(linear.interface_transfer(
            layers, 2 * np.pi / p, dist)) for p in periods])
        table[f"gain_x{dist:g}m"] = gain
        ax.semilogx(periods, gain, color=color, label=f"x = {dist:g} m")
    share = layers.salt / layers.total
    ax.axhline(share, color=plotting.INK, lw=0.8, ls="--")
    ax.text(0.12, share + 0.03, r"$H_s / H$", fontsize=7)
    for period, name in [(M2_PERIOD_DAYS, "M2"), (14.77, "spring-neap"),
                         (YEAR_DAYS, "annual")]:
        ax.axvline(period, color=plotting.MUTED, lw=0.6, ls=":")
        ax.text(period * 1.12, 1.02, name, rotation=90, fontsize=6.5,
                color=plotting.MUTED, va="top")
    ax.set_xlabel("forcing period (days)")
    ax.set_ylabel(r"interface amplitude / sea-level amplitude, $|G|$")
    ax.set_ylim(0, 1.05)
    plotting.legend_below(ax, ncol=2, title=r"$H_f = H_s$ = 10 m",
                          title_fontsize=7)
    plotting.panel_label(ax, "a")

    check = {}
    for color, period in zip(SERIES, ex.UNIFORM_PERIODS):
        run = ex.uniform_tide(amplitude=0.02, period=period, cycles=40)
        x = run["x"]
        z_hat = run["z_hat_re"] + 1j * run["z_hat_im"]
        _, z_theory = linear.periodic_response(
            layers, 2 * np.pi / period, x, 1.0, length=60.0)
        every = max(len(x) // 18, 1)
        bx.plot(x, np.abs(z_theory), color=color, lw=1.0,
                label=f"T = {period:g} d")
        bx.plot(x[::every], np.abs(z_hat[::every]) / 0.02, "o",
                color=color, mfc="white", mew=1.0, ms=4)
        check[f"x_m_T{period:g}"] = x
        check[f"numerical_T{period:g}"] = np.abs(z_hat) / 0.02
        check[f"theory_T{period:g}"] = np.abs(z_theory)
    bx.plot([], [], "o", color=plotting.INK, mfc="white", label="solver")
    bx.set_xlabel("x (m)")
    bx.set_ylabel(r"$|\hat\zeta| / A$")
    bx.set_xlim(0, 60)
    bx.set_ylim(0, 1.05)
    plotting.legend_below(bx, ncol=3)
    bx.set_title("closed at x = 60 m, A = 2 cm", fontsize=7,
                 color=plotting.MUTED)
    plotting.panel_label(bx, "b")

    plotting.save(fig, "fig03_transfer")
    write_csv("fig03a_transfer", table,
              "semi-infinite uniform layers, |zeta_hat / A|")
    for period in ex.UNIFORM_PERIODS:
        write_csv(f"fig03b_check_T{period:g}", {
            "x_m": check[f"x_m_T{period:g}"],
            "numerical": check[f"numerical_T{period:g}"],
            "theory": check[f"theory_T{period:g}"]},
            f"closed domain of 60 m, period {period:g} days")


if __name__ == "__main__":
    main()
