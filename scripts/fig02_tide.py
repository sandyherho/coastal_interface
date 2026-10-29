"""Figure 2: the tidal oscillation of the interface in the lens."""

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
    """Amplitudes, envelopes and phase of the tidal interface motion."""
    plotting.use_style()
    a = REFERENCE_AQUIFER
    run = ex.tidal_lens(amplitude=0.5, kind="flux")
    x = run["x"]
    amp = 0.5
    h_hat = run["h_hat_re"] + 1j * run["h_hat_im"]
    z_hat = run["z_hat_re"] + 1j * run["z_hat_im"]
    salt_share = (run["z_mean"] + a.base_depth) / (
        run["h_mean"] + a.base_depth)

    fig, axes = plt.subplots(1, 3, figsize=(plotting.DOUBLE, 2.45),
                             gridspec_kw={"wspace": 0.42})
    ax, bx, cx = axes

    near = x < 120
    ax.semilogy(x[near], run["h_amp"][near] / amp, color=plotting.FLUX,
                label="water table")
    ax.semilogy(x[near], run["z_amp"][near] / amp, color=plotting.HEAD,
                label="interface")
    ax.semilogy(x[near], run["gh_amp"][near] / amp, color=plotting.INK,
                lw=0.9, ls="--", label="Ghyben-Herzberg\n(instant sea level)")
    ax.axhline(1.0 + 1.0 / a.delta, color=plotting.MUTED, lw=0.6, ls=":")
    ax.text(118, (1 + 1 / a.delta) * 1.2, r"$1 + 1/\delta$", ha="right",
            fontsize=7, color=plotting.MUTED)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("oscillation amplitude / tidal amplitude")
    ax.set_ylim(1e-3, 120)
    ax.set_xlim(0, 120)
    plotting.legend_below(ax)
    plotting.panel_label(ax, "a")

    snaps_z = run["snap_z"]
    sea = 0.5 * np.cos(2 * np.pi * run["snap_t"] / ex.M2_PERIOD_DAYS)
    gh = core.ghyben_herzberg(run["snap_h"], sea[:, None], a.delta,
                              a.base_depth)
    view = x < 40
    bx.fill_between(x[view], gh[:, view].min(0), gh[:, view].max(0),
                    color=plotting.GRID, lw=0,
                    label="G-H range")
    bx.fill_between(x[view], snaps_z[:, view].min(0),
                    snaps_z[:, view].max(0), color=plotting.HEAD, lw=0,
                    alpha=0.85, label="model range")
    bx.plot(x[view], run["z_mean"][view], color=plotting.INK, lw=0.8,
            label="tidal mean")
    bx.set_xlim(0, 40)
    bx.set_ylim(-a.base_depth, 1.0)
    bx.set_xlabel("x (m)")
    bx.set_ylabel(r"interface elevation $\zeta$ (m)")
    plotting.legend_below(bx)
    plotting.panel_label(bx, "b")

    ratio = np.abs(z_hat) / np.abs(h_hat)
    lag = np.degrees(np.angle(z_hat / h_hat))
    mid = (x > 2) & (x < 70)
    cx.plot(x[mid], ratio[mid], color=plotting.HEAD,
            label=r"$|\hat\zeta| / |\hat h|$ (model)")
    cx.plot(x[mid], salt_share[mid], color=plotting.INK, lw=0.9, ls="--",
            label=r"$H_s/H$ of the mean lens")
    cx.set_xlabel("x (m)")
    cx.set_ylabel("interface / water-table amplitude")
    cx.set_ylim(0, 1.5)
    cx.set_xlim(0, 70)
    plotting.legend_below(cx)
    far = mid & (x > 20)
    twin_note = ("interface lags the water table\n"
                 f"by {-lag[far].max():.0f} to {-lag[far].min():.0f} deg "
                 "beyond 20 m")
    cx.text(68, 1.32, twin_note, fontsize=6.5, color=plotting.INK,
            ha="right")
    plotting.panel_label(cx, "c")

    plotting.save(fig, "fig02_tide")
    write_csv("fig02_tide", {
        "x_m": x, "h_amplitude_over_A": run["h_amp"] / amp,
        "zeta_amplitude_over_A": run["z_amp"] / amp,
        "ghyben_herzberg_amplitude_over_A": run["gh_amp"] / amp,
        "zeta_min_m": snaps_z.min(0), "zeta_max_m": snaps_z.max(0),
        "zeta_mean_m": run["z_mean"], "amplitude_ratio": ratio,
        "phase_zeta_minus_h_deg": lag, "salt_share": salt_share},
        "reference lens, flux-controlled, M2 tide of 0.5 m, last cycle")


if __name__ == "__main__":
    main()
