"""Figure 8: verification of the solver and of the closure."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(
    __file__))))

from coastal_interface import (experiments as ex, linear,  # noqa: E402
                               plotting, verification)
from coastal_interface.io_utils import write_csv  # noqa: E402
from coastal_interface.scenario import REFERENCE_AQUIFER  # noqa: E402

plt = plotting.plt


def sensitivity_rows():
    """Effective boundary coefficients for each sensitivity case."""
    rows = []
    for name, change in ex.sensitivity_cases().items():
        run = ex.tidal_lens(amplitude=0.5, kind="flux", **change)
        base = change.get("aquifer", REFERENCE_AQUIFER).base_depth
        rows.append((name, float(run["phi_e"]) * 4 * base / 0.25,
                     float(run["shift"]) / 0.25,
                     float(run["toe_mean"] - run["toe0"])))
    return rows


def _plain_log_ticks(axis, values):
    """Label a logarithmic x axis at the sampled values only."""
    axis.set_xticks(values)
    axis.set_xticks([], minor=True)
    axis.set_xticklabels([f"{v:g}" for v in values])


def main():
    """Convergence orders, rectification error and robustness."""
    plotting.use_style()
    fig, axes = plt.subplots(2, 2, figsize=(plotting.DOUBLE, 5.2),
                             gridspec_kw={"wspace": 0.35, "hspace": 0.75})
    (ax, bx), (cx, dx) = axes

    space = verification.space_convergence()
    cells = space["cells"]
    ax.loglog(cells, space["err_h"], "o-", color=plotting.FLUX,
              mfc="white", label="water table h")
    ax.loglog(cells, space["err_z"], "o-", color=plotting.HEAD,
              mfc="white", label=r"interface $\zeta$")
    ref = space["err_z"][0] * (cells / cells[0]) ** -2.0
    ax.loglog(cells, ref, color=plotting.INK, lw=0.8, ls="--",
              label="second order")
    _plain_log_ticks(ax, cells)
    ax.set_xlabel("number of cells")
    ax.set_ylabel("maximum error (m)")
    plotting.legend_below(ax, ncol=2)
    ax.set_title("space, steady manufactured solution", fontsize=7)
    plotting.panel_label(ax, "a")

    time = verification.time_convergence()
    steps = time["steps"]
    bx.loglog(steps, time["err"], "o-", color=plotting.FLUX, mfc="white",
              label="TR-BDF2")
    bx.loglog(steps, time["err"][0] * (steps / steps[0]) ** -2.0,
              color=plotting.INK, lw=0.8, ls="--", label="second order")
    _plain_log_ticks(bx, steps)
    bx.set_xlabel("time steps per day")
    bx.set_ylabel("maximum error (m)")
    plotting.legend_below(bx, ncol=2)
    bx.set_title("time, periodic manufactured solution", fontsize=7)
    plotting.panel_label(bx, "b")

    layers = linear.Layers.from_aquifer(REFERENCE_AQUIFER, 10.0, 10.0)
    amps = np.array(ex.UNIFORM_AMPLITUDES)
    omega = 2 * np.pi / ex.M2_PERIOD_DAYS
    err_h, err_z = [], []
    for amp in amps:
        run = ex.uniform_tide(amplitude=amp)
        h_th, z_th = linear.rectified_profiles(layers, omega, amp,
                                               run["x"], 60.0)
        err_h.append(abs(run["h_mean"][-1] / h_th[-1] - 1))
        err_z.append(abs(run["z_mean"][-1] / z_th[-1] - 1))
    cx.loglog(amps, err_h, "o-", color=plotting.FLUX, mfc="white",
              label=r"$\overline{h}$")
    cx.loglog(amps, err_z, "o-", color=plotting.HEAD, mfc="white",
              label=r"$\overline{\zeta}$")
    cx.loglog(amps, err_z[-1] * (amps / amps[-1]) ** 2, color=plotting.INK,
              lw=0.8, ls="--", label=r"$\propto A^2$")
    _plain_log_ticks(cx, amps)
    cx.set_xlabel("tidal amplitude A (m)")
    cx.set_ylabel("relative departure from the\nsecond-order means")
    plotting.legend_below(cx, ncol=3)
    cx.set_title("rectification, uniform layers", fontsize=7)
    plotting.panel_label(cx, "c")

    rows = sensitivity_rows()
    names = [r[0] for r in rows]
    c_phi = np.array([r[1] for r in rows])
    pos = np.arange(len(rows))[::-1]
    dx.plot(c_phi, pos, "o", color=plotting.FLUX, mfc="white")
    dx.axvline(1.0, color=plotting.INK, lw=0.8, ls="--")
    dx.axvline(c_phi[0], color=plotting.FLUX, lw=0.6, ls=":")
    dx.set_yticks(pos)
    dx.set_yticklabels(names)
    dx.set_xlabel(r"$4B\phi_e / A^2$ (A = 0.5 m, flux-controlled)")
    dx.set_title("effective salt-head rise", fontsize=7)
    plotting.panel_label(dx, "d")

    plotting.save(fig, "fig08_verification")
    write_csv("fig08a_space", {"cells": cells, "err_h_m": space["err_h"],
                               "err_zeta_m": space["err_z"]})
    write_csv("fig08b_time", {"steps_per_day": steps, "err_m": time["err"]})
    write_csv("fig08c_rectification", {
        "amplitude_m": amps, "rel_err_h": err_h, "rel_err_zeta": err_z})
    write_csv("fig08d_sensitivity", {
        "case": np.arange(len(rows)), "c_phi": c_phi,
        "shift_over_A2_per_m": [r[2] for r in rows],
        "toe_shift_m": [r[3] for r in rows]},
        "cases: " + "; ".join(f"{i} = {n}" for i, n in enumerate(names)))


if __name__ == "__main__":
    main()
