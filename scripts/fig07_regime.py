"""Figure 7: when does the tide matter next to the wave setup."""

import os
import sys

import numpy as np
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(
    __file__))))

from coastal_interface import experiments as ex, plotting  # noqa: E402
from coastal_interface.io_utils import write_csv  # noqa: E402
from coastal_interface.scenario import REFERENCE_AQUIFER  # noqa: E402

plt = plotting.plt

# Diverging map: setup-dominated (violet) to tide-dominated (amber),
# neutral gray where the two toe shifts are equal. These hues are kept
# apart from the flux/head colors used in the other figures.
DIVERGING = LinearSegmentedColormap.from_list(
    "tide_vs_setup", ["#6250d6", "#f2f1ee", "#eda100"])


def main():
    """Ratio of tidal to setup toe shift over amplitude and setup."""
    plotting.use_style()
    a = REFERENCE_AQUIFER
    c_phi, c_s = ex.closure_coefficients()
    # The closure was fitted for A up to 1 m; the map stays in that range.
    amps = np.linspace(0.05, 1.0, 80)
    setups = np.logspace(-3.0, np.log10(0.5), 80)

    fig, axes = plt.subplots(1, 2, figsize=(plotting.DOUBLE, 2.7),
                             sharey=True, gridspec_kw={"wspace": 0.1})
    for axis, kind, letter in zip(axes, ("flux", "head"), "ab"):
        tide = np.array([ex.closure_toe_shift(kind, amp, c_phi, c_s)
                         for amp in amps])
        setup = np.array([ex.setup_toe_shift(kind, s) for s in setups])
        ratio = tide[None, :] / setup[:, None]
        mesh = axis.pcolormesh(amps, setups, np.log10(ratio),
                               cmap=DIVERGING, vmin=-2, vmax=2,
                               shading="auto", rasterized=True)
        lines = axis.contour(amps, setups, np.log10(ratio),
                             levels=[-1, 0, 1], colors=plotting.INK,
                             linewidths=[0.6, 1.2, 0.6],
                             linestyles="solid")
        axis.clabel(lines, fmt={-1: "0.1", 0: "1", 1: "10"}, fontsize=6.5)
        axis.plot(amps, amps ** 2 / (4 * a.base_depth), color=plotting.INK,
                  lw=0.8, ls="--")
        axis.set_yscale("log")
        axis.set_ylim(setups[0], setups[-1])
        axis.set_xlabel("tidal amplitude A (m)")
        axis.set_title(f"{kind}-controlled", fontsize=8)
        axis.grid(False)
        plotting.panel_label(axis, letter)
        write_csv(f"fig07_ratio_{kind}", {
            "amplitude_m": np.repeat(amps[None, :], len(setups), 0).ravel(),
            "setup_m": np.repeat(setups[:, None], len(amps), 1).ravel(),
            "tide_over_setup_toe_shift": ratio.ravel()},
            f"closure with c_phi = {c_phi:.4f}, c_s = {c_s:.4f} 1/m")
    axes[0].set_ylabel(r"mean wave setup $\overline{\eta}_w$ (m)")
    axes[1].text(0.98, 0.0011, r"dashed: $\overline{\eta}_w = A^2/4B$",
                 ha="right", fontsize=6.5)
    bar = fig.colorbar(mesh, ax=axes, shrink=0.9, pad=0.02)
    bar.set_label(r"$\log_{10}$ (tidal / setup toe shift)")
    plotting.save(fig, "fig07_regime")


if __name__ == "__main__":
    main()
