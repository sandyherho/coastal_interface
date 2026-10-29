"""Animation 1: two tidal cycles at the shoreline end of the lens."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(
    __file__))))

from coastal_interface import anim, core, experiments as ex  # noqa: E402
from coastal_interface import plotting  # noqa: E402
from coastal_interface.scenario import REFERENCE_AQUIFER  # noqa: E402

plt = plotting.plt
AMPLITUDE = 1.0
X_MAX = 40.0


def main():
    """Model interface against the instantaneous Ghyben-Herzberg rule."""
    plotting.use_style()
    a = REFERENCE_AQUIFER
    run = ex.tidal_lens(amplitude=AMPLITUDE, kind="flux")
    view = run["x"] < X_MAX
    x = np.concatenate(([0.0], run["x"][view]))
    phase = np.arange(len(run["snap_t"])) / len(run["snap_t"])
    sea = AMPLITUDE * np.cos(2 * np.pi * run["snap_t"] / ex.M2_PERIOD_DAYS)
    h_all, z_all = run["snap_h"], run["snap_z"]

    fig = plt.figure(figsize=(6.4, 3.3))
    ax = fig.add_axes([0.09, 0.14, 0.62, 0.74])
    clock = fig.add_axes([0.78, 0.55, 0.2, 0.3])
    clock.plot(phase, sea, color=plotting.INK, lw=1.0)
    clock.set_xlabel("tidal phase", fontsize=7)
    clock.set_ylabel("sea level (m)", fontsize=7)
    clock.set_xlim(0, 1)
    dot, = clock.plot([], [], "o", color=plotting.HEAD, ms=5)
    fig.text(0.78, 0.36, "shading: model\n"
             "dashed: Ghyben-Herzberg\nwith the instantaneous\nsea level",
             fontsize=7, va="top", color=plotting.INK)
    fig.text(0.09, 0.93, f"M2 tide of {AMPLITUDE:g} m on the reference "
             "lens (flux-controlled)", fontsize=8)

    def update(frame):
        k = frame % len(phase)
        ax.clear()
        h = np.concatenate(([sea[k]], h_all[k][view]))
        z = np.concatenate(([sea[k]], z_all[k][view]))
        plotting.draw_lens(ax, x, h, z, a.base_depth, sea_level=sea[k],
                           x_max=X_MAX)
        gh = core.ghyben_herzberg(h, sea[k], a.delta, a.base_depth)
        ax.plot(x, gh, color=plotting.INK, lw=1.0, ls="--")
        ax.set_xlim(-0.06 * X_MAX, X_MAX)
        ax.set_ylim(-a.base_depth - 1.0, 1.6)
        ax.set_xlabel("distance from shoreline (m)")
        ax.set_ylabel("elevation (m)")
        ax.grid(False)
        dot.set_data([phase[k]], [sea[k]])
        return []

    anim.save_gif(fig, update, 2 * len(phase), "anim01_tide", fps=12)


if __name__ == "__main__":
    main()
