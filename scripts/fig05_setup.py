"""Figure 5: toe response to wave setup, steady and periodic."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(
    __file__))))

from coastal_interface import experiments as ex, plotting  # noqa: E402
from coastal_interface.io_utils import write_csv  # noqa: E402
from coastal_interface.scenario import (REFERENCE_AQUIFER,  # noqa: E402
                                        YEAR_DAYS)

plt = plotting.plt
COLORS = {"flux": plotting.FLUX, "head": plotting.HEAD}


def gains(kind):
    """Complex toe gain (m per m of setup) for every setup period."""
    out = []
    for period in ex.SETUP_PERIODS:
        run = ex.setup_cycle(kind=kind, period=period)
        out.append(complex(run["gain_re"], run["gain_im"]))
    return np.array(out)


def main():
    """Steady toe against setup, then gain and lag of a setup cycle."""
    plotting.use_style()
    a = REFERENCE_AQUIFER
    length = ex.REFERENCE_NUMERICS.length
    periods = np.array(ex.SETUP_PERIODS)

    fig, axes = plt.subplots(1, 3, figsize=(plotting.DOUBLE, 2.5),
                             gridspec_kw={"wspace": 0.42})
    ax, bx, cx = axes
    levels = np.linspace(0.0, 0.5, 51)
    for kind in ("flux", "head"):
        color = COLORS[kind]
        inland = ex.make_inland(kind, a, ex.REFERENCE_DISCHARGE, length)
        exact = [ex.quasi_static_toe(a, kind, s, ex.REFERENCE_DISCHARGE,
                                     inland.head, length) for s in levels]
        steady = ex.setup_steady(kind)
        ax.plot(levels, exact, color=color, lw=1.0)
        ax.plot(steady["level"], steady["toe"], "o", color=color,
                mfc="white", label=f"{kind}-controlled")
        write_csv(f"fig05a_steady_{kind}", {
            "setup_m": steady["level"], "toe_solver_m": steady["toe"],
            "toe_exact_m": steady["toe_exact"]},
            "steady toe for a constant setup")
    ax.plot([], [], color=plotting.INK, lw=1.0, label="Strack-Glover")
    ax.set_xlabel(r"mean wave setup $\overline{\eta}_w$ (m)")
    ax.set_ylabel(r"toe position $x_T$ (m)")
    plotting.legend_below(ax)
    plotting.panel_label(ax, "a")

    for kind in ("flux", "head"):
        color = COLORS[kind]
        inland = ex.make_inland(kind, a, ex.REFERENCE_DISCHARGE, length)
        gain = gains(kind)
        static = ex.quasi_static_gain(a, kind, ex.REFERENCE_DISCHARGE,
                                      inland.head, length)
        lag = -np.angle(gain) / (2 * np.pi) * periods
        bx.loglog(periods, np.abs(gain), "o-", color=color, mfc="white",
                  lw=1.0, label=f"{kind}-controlled")
        bx.axhline(static, color=color, lw=0.8, ls="--")
        cx.semilogx(periods, lag, "o-", color=color, mfc="white", lw=1.0,
                    label=f"{kind}-controlled")
        write_csv(f"fig05bc_cycle_{kind}", {
            "period_days": periods, "gain_m_per_m": np.abs(gain),
            "lag_days": lag,
            "quasi_static_gain_m_per_m": np.full(len(periods), static)},
            "toe response to a 5 cm sinusoidal setup")
    bx.plot([], [], color=plotting.INK, lw=0.8, ls="--",
            label="quasi-static")
    for axis in (bx, cx):
        axis.axvline(YEAR_DAYS, color=plotting.MUTED, lw=0.6, ls=":")
        axis.set_xlabel("setup period (days)")
    bx.set_ylabel("toe excursion per meter of setup (m/m)")
    bx.set_ylim(1.0, 150)
    plotting.legend_below(bx)
    plotting.panel_label(bx, "b")
    cx.axhline(0.0, color=plotting.MUTED, lw=0.6)
    cx.text(YEAR_DAYS * 1.12, 170, "annual", rotation=90, fontsize=6.5,
            color=plotting.MUTED, va="top")
    cx.set_ylabel("toe lag behind setup (days)")
    plotting.legend_below(cx)
    plotting.panel_label(cx, "c")

    plotting.save(fig, "fig05_setup")


if __name__ == "__main__":
    main()
