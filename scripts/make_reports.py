"""Plain-text reports: verification, key numbers, sensitivity, open items."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(
    __file__))))

from coastal_interface import (core, experiments as ex, linear,  # noqa
                               verification)
from coastal_interface.io_utils import write_report  # noqa: E402
from coastal_interface.scenario import (M2_PERIOD_DAYS,  # noqa: E402
                                        REFERENCE_AQUIFER, YEAR_DAYS)

A = REFERENCE_AQUIFER
LENGTH = ex.REFERENCE_NUMERICS.length
Q = ex.REFERENCE_DISCHARGE


def fmt(value, digits=4):
    """Compact number formatting for the tables."""
    return f"{value:.{digits}g}"


def verification_report():
    """Every check of the numerics against exact results."""
    lines = ["Verification of the two-layer sharp-interface solver",
             "=" * 54, ""]

    space = verification.space_convergence()
    orders_h = verification.observed_order(space["cells"], space["err_h"])
    orders_z = verification.observed_order(space["cells"], space["err_z"])
    lines += ["1. Spatial order, steady manufactured solution "
              "(graded grid, cells halved)",
              "   cells    max|h error|   order   max|zeta error|  order"]
    for i, n in enumerate(space["cells"]):
        oh = fmt(orders_h[i - 1], 3) if i else "-"
        oz = fmt(orders_z[i - 1], 3) if i else "-"
        lines.append(f"   {n:5d}    {space['err_h'][i]:.3e}    {oh:>5}   "
                     f"{space['err_z'][i]:.3e}      {oz:>5}")

    time = verification.time_convergence()
    orders_t = verification.observed_order(time["steps"], time["err"])
    lines += ["", "2. Temporal order, TR-BDF2, periodic manufactured "
              "solution over one day", "   steps/day   max error    order"]
    for i, n in enumerate(time["steps"]):
        ot = fmt(orders_t[i - 1], 3) if i else "-"
        lines.append(f"   {n:9d}   {time['err'][i]:.3e}   {ot:>5}")

    lines += ["", "3. Discrete steady lens against Strack-Glover "
              "(reference grid)", "   case                    toe solver (m)"
              "   toe exact (m)   difference"]
    grid = ex.make_grid()
    for discharge in (1.0, 2.0, 3.0):
        inland = ex.make_inland("flux", A, discharge, LENGTH)
        y = ex.initial_lens(grid, A, inland, discharge)
        toe = core.toe_position(grid.centers, y[0::2], y[1::2], A.delta,
                                A.base_depth)
        exact = core.toe_length_flux(A, discharge)
        lines.append(f"   flux, Q = {discharge:g} m2/d          {toe:12.6f}"
                     f"   {exact:12.6f}   {toe - exact:+.2e}")
    for kind in ("flux", "head"):
        st = ex.setup_steady(kind)
        for lev, num, exa in zip(st["level"], st["toe"], st["toe_exact"]):
            lines.append(f"   {kind}, setup {lev:.1f} m        "
                         f"{num:12.6f}   {exa:12.6f}   {num - exa:+.2e}")

    lines += ["", "4. Linear tidal response, uniform layers 10 m + 10 m, "
              "closed at 60 m, A = 2 cm",
              "   period (d)   max |zeta_hat - theory| / A   "
              "max |h_hat - theory| / A"]
    layers = linear.Layers.from_aquifer(A, 10.0, 10.0)
    for period in ex.UNIFORM_PERIODS:
        run = ex.uniform_tide(amplitude=0.02, period=period, cycles=40)
        h_th, z_th = linear.periodic_response(
            layers, 2 * np.pi / period, run["x"], 0.02, 60.0)
        ez = np.max(np.abs(run["z_hat_re"] + 1j * run["z_hat_im"]
                           - z_th)) / 0.02
        eh = np.max(np.abs(run["h_hat_re"] + 1j * run["h_hat_im"]
                           - h_th)) / 0.02
        lines.append(f"   {period:10.4g}   {ez:.3e}                    "
                     f"{eh:.3e}")

    lines += ["", "5. Rectified means, uniform layers, M2, at the closed end "
              "x = 60 m",
              "   theory (60 m) integrates the exact tidal fluxes of the "
              "closed domain; closed form is the",
              "   semi-infinite result A^2/[4(H + delta H_s)] and "
              "A^2/[4(H + delta H_s)] + A^2/(4 H_s)",
              "   A (m)  quantity   solver         theory (60 m)  rel. diff."
              "   closed form"]
    total = layers.total + layers.delta * layers.salt
    omega = 2 * np.pi / M2_PERIOD_DAYS
    for amp in ex.UNIFORM_AMPLITUDES:
        run = ex.uniform_tide(amplitude=amp)
        h_th, z_th = linear.rectified_profiles(layers, omega, amp, run["x"],
                                               60.0)
        closed_h = amp ** 2 / (4 * total)
        closed_z = closed_h + amp ** 2 / (4 * layers.salt)
        for name, num, th, cf in (("h mean", run["h_mean"][-1], h_th[-1],
                                   closed_h),
                                  ("zeta mean", run["z_mean"][-1], z_th[-1],
                                   closed_z)):
            lines.append(f"   {amp:4.2f}   {name:9s}  {num:.6e}   {th:.6e}"
                         f"   {num / th - 1:+.1e}    {cf:.6e}")

    lines += ["", "6. Volume budgets of the tidal lens runs (boundary "
              "exchange integrated with the TR-BDF2 weights)",
              "   run                                    fresh residual (m2)"
              "   salt residual (m2)   step halvings"]
    labels = [f"{kind}, A = {amp:g} m" for kind in ("flux", "head")
              for amp in ex.AMPLITUDES]
    labels += [f"flux, A = 0.5 m, {name}"
               for name in ex.sensitivity_cases() if name != "reference"]
    for label, job in zip(labels, ex.tidal_jobs()):
        run = ex.tidal_lens(**job)
        lines.append(f"   {label:40s} {float(run['budget_fresh']):+.2e}"
                     f"            {float(run['budget_salt']):+.2e}"
                     f"            {int(run['halvings'])}")

    lines += ["", "7. Numerical sensitivity of the effective boundary "
              "(A = 0.5 m, flux-controlled)",
              "   case                  4B phi_e / A^2   s / A^2 (1/m)   "
              "toe shift (m)"]
    for name in ("reference", "grid halved", "64 steps per tide"):
        change = ex.sensitivity_cases()[name]
        run = ex.tidal_lens(amplitude=0.5, kind="flux", **change)
        lines.append(f"   {name:20s}  {float(run['phi_e']) * 80 / 0.25:.5f}"
                     f"          {float(run['shift']) / 0.25:.5f}"
                     f"          {float(run['toe_mean'] - run['toe0']):.5f}")
    return lines


def key_numbers():
    """Collect the numbers quoted in the README and the paper."""
    lines = ["Key numbers", "=" * 11, ""]
    layers = linear.Layers.from_aquifer(A, 10.0, 10.0)
    d0, d1 = linear.mode_diffusivities(layers)
    lo0, lo1 = linear.leading_order_diffusivities(layers)
    lines += ["Reference aquifer: K = 30 m/d, n = 0.30, delta = 0.025, "
              "B = 20 m, Q = 2 m2/d per m of shoreline, domain 300 m.",
              "", "Modes of uniform layers H_f = H_s = 10 m:",
              f"   D_0 = {d0:.2f} m2/d (K H / n = {lo0:.1f}),  "
              f"D_1 = {d1:.3f} m2/d (delta K H_f H_s / (n H) = {lo1:.2f})",
              f"   D_1 / D_0 = {d1 / d0:.5f}  (bound delta/4 = "
              f"{A.delta / 4:.5f})"]
    for period, name in ((M2_PERIOD_DAYS, "M2"), (14.77, "spring-neap"),
                         (YEAR_DAYS, "annual")):
        l0, l1 = linear.skin_depths(layers, 2 * np.pi / period)
        lines.append(f"   decay lengths at {name:11s}: L_0 = {l0:7.1f} m,"
                     f"  L_1 = {l1:6.2f} m")

    run = ex.tidal_lens(amplitude=0.5, kind="flux")
    lines += ["", "Tidal oscillation in the lens (M2, A = 0.5 m, "
              "flux-controlled):",
              f"   largest interface amplitude / A = "
              f"{float(np.max(run['z_amp'])) / 0.5:.3f}",
              f"   largest instantaneous Ghyben-Herzberg amplitude / A = "
              f"{float(np.max(run['gh_amp'])) / 0.5:.1f}"]

    rows = ex.lens_closure_table()
    c_phi, c_s = ex.closure_coefficients(rows)
    lines += ["", "Rectified means of uniform layers (exact, second order):",
              "   mean water-table rise  A^2 / [4 (H + delta H_s)]",
              "   mean interface rise    A^2 / [4 (H + delta H_s)] "
              "+ A^2 / (4 H_s)",
              "   freshwater thickness   - A^2 / (4 H_s); independent of "
              "K, n and frequency",
              "", "Lens: effective boundary fitted outside the tidal layer",
              f"   phi_e = c_phi A^2 / (4B) with c_phi = {c_phi:.4f}",
              f"   s     = c_s A^2        with c_s   = {c_s:.4f} 1/m",
              "   A (m)   kind   phi_e (mm)   s (m)    toe shift (m)   "
              "closure (m)"]
    for kind in ("flux", "head"):
        for amp, phi, s, shift in rows[kind]:
            pred = ex.closure_toe_shift(kind, amp, c_phi, c_s)
            lines.append(f"   {amp:4.2f}    {kind:4s}   {1e3 * phi:8.3f}"
                         f"   {s:6.4f}   {shift:9.4f}       {pred:9.4f}")

    lines += ["", "Wave setup (steady):"]
    for kind in ("flux", "head"):
        shift = ex.setup_toe_shift(kind, 0.1)
        lines.append(f"   {kind}-controlled: toe moves {shift:.2f} m "
                     "landward for 0.1 m of mean setup")
    lines += ["", "Tide against setup (toe shifts, closure):"]
    for kind in ("flux", "head"):
        tide = ex.closure_toe_shift(kind, 1.0, c_phi, c_s)
        setups = np.logspace(-4, 0, 4001)
        shifts = np.array([ex.setup_toe_shift(kind, s) for s in setups])
        equal = float(np.interp(tide, shifts, setups))
        lines.append(f"   {kind}-controlled: a 1 m M2 tide moves the toe as"
                     f" far as a mean setup of {100 * equal:.2f} cm")

    lines += ["", "Setup cycles (5 cm amplitude), toe excursion per meter "
              "of setup:", "   period (d)   flux gain   flux lag (d)   "
              "head gain   head lag (d)"]
    for period in ex.SETUP_PERIODS:
        vals = []
        for kind in ("flux", "head"):
            r = ex.setup_cycle(kind=kind, period=period)
            g = complex(r["gain_re"], r["gain_im"])
            vals += [abs(g), -np.angle(g) / (2 * np.pi) * period]
        lines.append(f"   {period:10.1f}   {vals[0]:9.2f}   {vals[1]:12.1f}"
                     f"   {vals[2]:9.2f}   {vals[3]:12.1f}")
    for kind in ("flux", "head"):
        inland = ex.make_inland(kind, A, Q, LENGTH)
        static = ex.quasi_static_gain(A, kind, Q, inland.head, LENGTH)
        lines.append(f"   quasi-static gain, {kind}: {static:.2f} m/m")

    lines += ["", "Seasonal run (setup 0.2 m mean, 0.15 m annual amplitude, "
              "M2 0.5 m), last year:"]
    for kind in ("flux", "head"):
        r = ex.seasonal(kind=kind, tide_amplitude=0.5)
        last = r["t"] > r["t"][-1] - YEAR_DAYS
        lines.append(
            f"   {kind}: toe range {np.ptp(r['toe'][last]):.2f} m "
            f"(quasi-static {np.ptp(r['toe_quasi_static'][last]):.2f} m),"
            f" mean {np.mean(r['toe'][last]):.2f} m "
            f"(quasi-static {np.mean(r['toe_quasi_static'][last]):.2f} m,"
            f" untided start {float(r['toe0']):.2f} m)")
    return lines


def sensitivity_report():
    """Single-parameter changes of the reference tidal run."""
    lines = ["Sensitivity of the tidal effective boundary (A = 0.5 m, "
             "flux-controlled)", "=" * 70, "",
             "Base depth and conductivity are changed together with the "
             "discharge so that the untided toe stays at 76.9 m.", "",
             "case                  4B phi_e / A^2   s / A^2 (1/m)   "
             "toe shift (m)   phi spread in fit window (mm)"]
    for name, change in ex.sensitivity_cases().items():
        run = ex.tidal_lens(amplitude=0.5, kind="flux", **change)
        base = change.get("aquifer", A).base_depth
        lines.append(f"{name:20s}  {float(run['phi_e']) * 4 * base / 0.25:.4f}"
                     f"           {float(run['shift']) / 0.25:.4f}"
                     f"          {float(run['toe_mean'] - run['toe0']):.4f}"
                     f"          {1e3 * float(run['phi_spread']):.4f}")
    lines += ["", "The thin-lens theory predicts 4B phi_e / A^2 = 1 for every "
              "case; the departure measures the effect of the finite",
              "freshwater thickness inside the tidal layer. The shoreline "
              "shift s has no theory here and is reported as fitted."]
    return lines


def open_items():
    """List what is taken from earlier work, what is new, what is open."""
    return [
        "Provenance of the results and open items",
        "=" * 40, "",
        "Taken from earlier work (to be cited; references to be checked "
        "against the original papers before submission):",
        "  - Ghyben-Herzberg relation (Badon Ghyben 1888; Herzberg 1901).",
        "  - Dupuit sharp-interface (two-fluid) formulation and the steady",
        "    single-potential lens (Strack 1976, Water Resources Research;",
        "    Bear 1972, Dynamics of Fluids in Porous Media).",
        "  - Linear tidal propagation in coastal aquifers (Jacob 1950;",
        "    Ferris 1951) and the freshwater-only tidal overheight",
        "    (Philip 1973; Nielsen 1990, Water Resources Research).",
        "  - Flux- versus head-controlled responses to sea-level change",
        "    (Werner and Simmons 2009, Ground Water) and seawater-intrusion",
        "    overshoot after a step change (Watson et al. 2010, Water",
        "    Resources Research; Morgan 2014, PhD thesis, Flinders",
        "    University, sand-tank tests).",
        "  - Wave setup and run-up raising coastal water tables (Kang and",
        "    Nielsen, ICCE proceedings).",
        "  - TR-BDF2 (Bank et al. 1985; Hosea and Shampine 1996).",
        "", "Believed new here (a literature search is still needed before",
        "claiming priority):",
        "  - The exact barotropic/baroclinic split of the linearized",
        "    two-layer aquifer, with D_1/D_0 <= delta/4.",
        "  - The Wronskian identity for two-mode diffusion, which gives the",
        "    rectified means in closed form, independent of K, n and omega.",
        "  - The result that a tide thins the freshwater layer by A^2/(4H_s)",
        "    and raises the effective saltwater head by A^2/(4B) in the",
        "    thin-lens limit, so that its effect on the toe equals a mean",
        "    sea-level rise of millimeters.",
        "  - The two-parameter effective-boundary closure (phi_e, s) that",
        "    carries the tidal layer to the far-field lens.",
        "  - The toe frequency response to periodic setup, including the",
        "    gain above the quasi-static value in flux-controlled aquifers.",
        "", "Open items:",
        "  - The shoreline shift s is fitted, not derived.",
        "  - The Dupuit assumption fails within a few aquifer thicknesses of",
        "    the shore; the tidal interface layer (L_1 of order 1 m at M2)",
        "    lies well inside that zone, so tidal-band results there are",
        "    qualitative. The mean far-field results rest on the mean lens,",
        "    whose horizontal scale (77 m) exceeds B (20 m) by about four.",
        "  - No mixing zone, no seepage face, no wave run-up infiltration,",
        "    no upper saline plume, no beach slope, no recharge.",
        "  - Equal viscosities and specific yield equal to the porosity that",
        "    the interface sweeps.",
        "  - Parameter values are illustrative ranges for clean sands, not",
        "    measurements of any site.",
    ]


def main():
    """Write the four reports."""
    write_report("verification", verification_report())
    write_report("key_numbers", key_numbers())
    write_report("sensitivity", sensitivity_report())
    write_report("open_items", open_items())


if __name__ == "__main__":
    main()
