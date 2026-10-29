"""Figure 4: the rectified (tidal-mean) response, theory and lens."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(
    __file__))))

from coastal_interface import experiments as ex, linear, plotting  # noqa
from coastal_interface.io_utils import write_csv  # noqa: E402
from coastal_interface.scenario import (M2_PERIOD_DAYS,  # noqa: E402
                                        REFERENCE_AQUIFER)

plt = plotting.plt


def main():
    """Uniform-layer theory against the solver, then the lens."""
    plotting.use_style()
    a = REFERENCE_AQUIFER
    layers = linear.Layers.from_aquifer(a, 10.0, 10.0)
    omega = 2 * np.pi / M2_PERIOD_DAYS
    fig, axes = plt.subplots(2, 2, figsize=(plotting.DOUBLE, 5.4),
                             gridspec_kw={"wspace": 0.3, "hspace": 0.62})
    (ax, bx), (cx, dx) = axes

    run = ex.uniform_tide(amplitude=0.2)
    x = run["x"]
    h_th, z_th = linear.rectified_profiles(layers, omega, 0.2, x, 60.0)
    every = max(len(x) // 25, 1)
    ax.plot(x, 1e3 * h_th, color=plotting.FLUX, lw=1.0,
            label=r"$\overline{h}$ theory")
    ax.plot(x, 1e3 * z_th, color=plotting.HEAD, lw=1.0,
            label=r"$\overline{\zeta}$ theory")
    ax.plot(x[::every], 1e3 * run["h_mean"][::every], "o",
            color=plotting.FLUX, mfc="white", ms=3.5, label="solver")
    ax.plot(x[::every], 1e3 * run["z_mean"][::every], "o",
            color=plotting.HEAD, mfc="white", ms=3.5)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("tidal-mean rise (mm)")
    ax.set_xlim(0, 60)
    ax.set_ylim(-0.1, 3.4)
    plotting.legend_below(ax, ncol=3)
    ax.text(30, 0.15, r"$H_f = H_s$ = 10 m, A = 0.2 m", ha="center",
            fontsize=6.5, color=plotting.MUTED)
    plotting.panel_label(ax, "a")

    amps = np.array(ex.UNIFORM_AMPLITUDES)
    far_h, far_z = [], []
    for amp in amps:
        r = ex.uniform_tide(amplitude=amp)
        far_h.append(r["h_mean"][-1])
        far_z.append(r["z_mean"][-1])
    grid_a = np.logspace(np.log10(0.03), np.log10(0.6), 50)
    total = layers.total + layers.delta * layers.salt
    bx.loglog(grid_a, grid_a ** 2 / (4 * total), color=plotting.FLUX,
              lw=1.0, label=r"$A^2 / 4(H + \delta H_s)$")
    bx.loglog(grid_a, grid_a ** 2 / (4 * total)
              + grid_a ** 2 / (4 * layers.salt), color=plotting.HEAD,
              lw=1.0, label=r"$+\, A^2 / 4H_s$")
    bx.loglog(amps, far_h, "o", color=plotting.FLUX, mfc="white")
    bx.loglog(amps, far_z, "o", color=plotting.HEAD, mfc="white")
    bx.set_xlabel("tidal amplitude A (m)")
    bx.set_ylabel("far-field mean rise (m)")
    bx.set_xticks([0.05, 0.1, 0.2, 0.4])
    bx.set_xticks([], minor=True)
    bx.set_xticklabels(["0.05", "0.1", "0.2", "0.4"])
    plotting.legend_below(bx, ncol=2)
    plotting.panel_label(bx, "b")

    rows = ex.lens_closure_table()
    c_phi, c_s = ex.closure_coefficients(rows)
    fine = np.linspace(0.0, 1.05, 60)
    cx.plot(fine, 1e3 * fine ** 2 / (4 * a.base_depth), color=plotting.INK,
            lw=0.9, ls="--", label=r"$A^2/4B$ (thin-lens limit)")
    for kind, color, mark, size in (("flux", plotting.FLUX, "o", 6),
                                    ("head", plotting.HEAD, "s", 3)):
        cx.plot(rows[kind][:, 0], 1e3 * rows[kind][:, 1], mark,
                color=color, mfc="white", ms=size,
                label=f"{kind}-controlled lens")
    cx.set_xlabel("tidal amplitude A (m)")
    cx.set_ylabel(r"effective salt-head rise $\phi_e$ (mm)")
    cx.set_xlim(0, 1.05)
    cx.set_ylim(bottom=0)
    plotting.legend_below(cx)
    plotting.panel_label(cx, "c")

    for kind, color in (("flux", plotting.FLUX), ("head", plotting.HEAD)):
        pred = [ex.closure_toe_shift(kind, amp, c_phi, c_s)
                for amp in fine]
        pred_phi = [ex.closure_toe_shift(kind, amp, c_phi, c_s, False)
                    for amp in fine]
        dx.plot(fine, pred, color=color, lw=1.0)
        dx.plot(fine, pred_phi, color=color, lw=0.8, ls=":")
        dx.plot(rows[kind][:, 0], rows[kind][:, 3], "o", color=color,
                mfc="white", label=f"{kind}-controlled")
    dx.plot([], [], color=plotting.INK, lw=1.0,
            label=r"closure, $\phi_e$ and $s$")
    dx.plot([], [], color=plotting.INK, lw=0.8, ls=":",
            label=r"closure, $\phi_e$ only")
    dx.set_xlabel("tidal amplitude A (m)")
    dx.set_ylabel("landward toe shift (m)")
    dx.set_xlim(0, 1.05)
    dx.set_ylim(bottom=0)
    plotting.legend_below(dx, ncol=2)
    plotting.panel_label(dx, "d")

    plotting.save(fig, "fig04_rectification")
    write_csv("fig04a_uniform_means", {
        "x_m": x, "h_mean_m": run["h_mean"], "zeta_mean_m": run["z_mean"],
        "h_theory_m": h_th, "zeta_theory_m": z_th},
        "uniform layers 10 m + 10 m, closed at 60 m, A = 0.2 m, M2")
    write_csv("fig04b_far_field", {
        "amplitude_m": amps, "h_far_m": far_h, "zeta_far_m": far_z,
        "h_theory_m": amps ** 2 / (4 * total),
        "zeta_theory_m": amps ** 2 / (4 * total) + amps ** 2 / (
            4 * layers.salt)}, "values at the closed end, x = 60 m")
    for kind in ("flux", "head"):
        write_csv(f"fig04cd_lens_{kind}", {
            "amplitude_m": rows[kind][:, 0], "phi_e_m": rows[kind][:, 1],
            "shift_s_m": rows[kind][:, 2], "toe_shift_m": rows[kind][:, 3]},
            f"{kind}-controlled lens; fitted c_phi = {c_phi:.4f}, "
            f"c_s = {c_s:.4f} 1/m")


if __name__ == "__main__":
    main()
