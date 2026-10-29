"""Numerical experiments behind the figures and reports.

Every experiment is a plain function of scalar parameters that returns
a dict of arrays, so that :func:`io_utils.cached` can store it and the
plotting scripts never rerun a simulation they already have.
"""

import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from . import core
from .io_utils import cached
from .scenario import (M2_PERIOD_DAYS, REFERENCE_AQUIFER, REFERENCE_NUMERICS,
                       Inland, Numerics, SeaForcing)
from .solver import (BoundaryBudget, Grid, TRBDF2, TwoLayerAquifer,
                     interleave, steady_state)

REFERENCE_DISCHARGE = 2.0


# -- shared set-up ---------------------------------------------------------
def make_grid(numerics=REFERENCE_NUMERICS):
    """Graded grid of the reference numerics."""
    return Grid(numerics.length, numerics.dx_min, numerics.dx_max,
                numerics.growth)


def make_inland(kind, aquifer, discharge, length):
    """Inland condition that sustains ``discharge`` at mean sea level.

    For ``"head"`` the head is that of the steady Strack-Glover lens at
    ``x = length``, so the flux- and head-controlled aquifers start from
    the same lens.
    """
    if kind == "flux":
        return Inland("flux", discharge=discharge)
    head = core.steady_lens(aquifer, discharge, np.array([length]))[0][0]
    return Inland("head", head=float(head))


def initial_lens(grid, aquifer, inland, discharge, sea_level=0.0):
    """Discrete steady lens for a constant sea level."""
    h, zeta = core.steady_lens(aquifer, discharge, grid.centers, sea_level)
    model = TwoLayerAquifer(grid, aquifer, lambda t: (sea_level, sea_level),
                            inland)
    return steady_state(model, interleave(h, zeta))


def run_parallel(function, jobs, workers=None):
    """Evaluate ``function(**job)`` for each job on several processes."""
    workers = workers or min(len(jobs), os.cpu_count() or 1)
    if workers <= 1:
        return [function(**job) for job in jobs]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(function, **job) for job in jobs]
        return [f.result() for f in futures]


def _beach(sea):
    """Shoreline values for a beach face: both unknowns equal the sea."""
    def shore(t):
        level = float(sea.level(t))
        return level, level
    return shore


def _closure_window(aquifer, omega, x_toe):
    """Part of the lens outside the tidal layer and short of the toe."""
    d0 = aquifer.conductivity * aquifer.base_depth / aquifer.porosity
    return 2.0 * np.sqrt(2.0 * d0 / omega), 0.85 * x_toe


def effective_boundary(x, h_mean, z_mean, aquifer, discharge, window):
    """Fit the far-field mean lens to a shifted Strack-Glover lens.

    Outside the tidal layer the saltwater is at rest in the mean, with a
    uniform head ``phi_e``, and the Kirchhoff variable
    ``u = (h - phi)^2`` grows as ``2 delta Q (x - s) / (K (1 + delta))``.
    The two numbers ``phi_e`` (an effective rise of the saltwater head
    at the shore) and ``s`` (an effective landward shift of the
    shoreline) summarize what the tide does to the rest of the lens.
    """
    d, k = aquifer.delta, aquifer.conductivity
    phi = (h_mean + d * z_mean) / (1.0 + d)
    sel = (x > window[0]) & (x < window[1])
    slope = 2.0 * d * discharge / (k * (1.0 + d))
    shift = x[sel] - (h_mean[sel] - phi[sel]) ** 2 / slope
    return float(np.mean(phi[sel])), float(np.mean(shift)), float(
        np.std(phi[sel]))


# -- tides on the lens -----------------------------------------------------
def _tidal_lens(amplitude, kind, period, aquifer, numerics, days,
                discharge, steps_per_tide):
    grid = make_grid(numerics)
    inland = make_inland(kind, aquifer, discharge, numerics.length)
    y0 = initial_lens(grid, aquifer, inland, discharge)
    sea = SeaForcing(tide_amplitude=amplitude, tide_period=period,
                     ramp=4.0 * period)
    model = TwoLayerAquifer(grid, aquifer, _beach(sea), inland)
    budget = BoundaryBudget(model)
    stepper = TRBDF2(model, newton_tol=1e-10, on_step=budget)
    d, b = aquifer.delta, aquifer.base_depth
    x = grid.centers

    def toe(y):
        return core.toe_position(x, y[0::2], y[1::2], d, b)

    toe0 = toe(y0)
    vf0, vs0 = model.volumes(y0)
    cycles = int(np.ceil(days / period))
    dt = period / steps_per_tide

    sums = np.zeros(3)
    series = []
    snapshots = []

    def observe(t, y):
        k = int(round(t / dt))
        if k == 0:
            return
        vf, vs = model.volumes(y)
        sums[:] += (toe(y), vs, vf)
        if k > (cycles - 1) * steps_per_tide:
            snapshots.append(y.copy())
        if k % steps_per_tide == 0:
            mean = sums / steps_per_tide
            series.append((t / period, mean[0] - toe0, mean[1] - vs0,
                           mean[2] - vf0))
            sums[:] = 0.0

    y_end = stepper.run(y0, 0.0, cycles * period, dt, observe=observe)
    snaps = np.array(snapshots)
    h_s, z_s = snaps[:, 0::2], snaps[:, 1::2]
    h_mean, z_mean = h_s.mean(axis=0), z_s.mean(axis=0)
    phase = 2.0 * np.pi * np.arange(1, steps_per_tide + 1) / steps_per_tide
    basis = np.exp(-1j * phase)[:, None]
    h_hat = 2.0 * np.mean(h_s * basis, axis=0)
    z_hat = 2.0 * np.mean(z_s * basis, axis=0)
    times = (cycles - 1) * period + phase / (2.0 * np.pi) * period
    eta = sea.level(times)[:, None]
    gh = core.ghyben_herzberg(h_s, eta, d, b)

    mean_q = np.mean([-model.face_fluxes(t, y)[0][-1]
                      for t, y in zip(times, snaps)])
    window = _closure_window(aquifer, 2.0 * np.pi / period, toe0)
    phi_e, shift, phi_spread = effective_boundary(
        x, h_mean, z_mean, aquifer, mean_q, window)
    vf, vs = model.volumes(y_end)
    vf0, vs0 = model.volumes(y0)
    series = np.array(series)
    return {
        "x": x, "h0": y0[0::2], "z0": y0[1::2],
        "h_mean": h_mean, "z_mean": z_mean,
        "h_amp": 0.5 * np.ptp(h_s, axis=0), "z_amp": 0.5 * np.ptp(z_s, axis=0),
        "gh_amp": 0.5 * np.ptp(gh, axis=0),
        "h_hat_re": h_hat.real, "h_hat_im": h_hat.imag,
        "z_hat_re": z_hat.real, "z_hat_im": z_hat.imag,
        "snap_h": h_s, "snap_z": z_s, "snap_t": times,
        "cycle": series[:, 0], "toe_shift": series[:, 1],
        "salt_volume": series[:, 2], "fresh_volume": series[:, 3],
        "toe0": toe0,
        "toe_mean": core.toe_position(x, h_mean, z_mean, d, b),
        "phi_e": phi_e, "shift": shift, "phi_spread": phi_spread,
        "mean_discharge": mean_q, "window": np.array(window),
        "budget_fresh": (vf - vf0) - budget.fresh_in,
        "budget_salt": (vs - vs0) - budget.salt_in,
        "halvings": stepper.halvings,
    }


def tidal_lens(amplitude=0.5, kind="flux", period=M2_PERIOD_DAYS,
               aquifer=REFERENCE_AQUIFER, numerics=REFERENCE_NUMERICS,
               days=1000.0, discharge=REFERENCE_DISCHARGE,
               steps_per_tide=32):
    """Lens under a single tidal constituent until the mean settles.

    Returns cycle-averaged toe position and layer volumes (relative to
    the untided lens), the last cycle of profiles, the tidally averaged
    lens and its effective-boundary fit, and the volume-budget residuals.
    """
    return cached("tidal_lens", _tidal_lens, amplitude=amplitude,
                  kind=kind, period=period, aquifer=aquifer,
                  numerics=numerics, days=days, discharge=discharge,
                  steps_per_tide=steps_per_tide)


def predicted_toe(aquifer, kind, phi_e, shift, discharge, head, length):
    """Toe of a Strack-Glover lens with an effective boundary.

    The lens starts at ``x = shift`` with saltwater head ``phi_e``; for a
    head-controlled aquifer the discharge follows from the inland head
    over the shortened length ``length - shift``.
    """
    if kind == "head":
        discharge = core.discharge_for_head(aquifer, head, length - shift,
                                            phi_e)
    return shift + core.toe_length_flux(aquifer, discharge, phi_e)


# -- tides on uniform layers -----------------------------------------------
def _uniform_tide(amplitude, period, fresh, salt, length, cycles,
                  steps_per_tide, aquifer):
    grid = Grid(length, 0.02, 0.5, 1.03)
    z0 = -aquifer.base_depth + salt
    h0 = z0 + fresh
    sea = SeaForcing(tide_amplitude=amplitude, tide_period=period,
                     ramp=2.0 * period)

    def shore(t):
        level = float(sea.level(t))
        return h0 + level, z0 + level

    model = TwoLayerAquifer(grid, aquifer, shore, Inland("flux", 0.0))
    stepper = TRBDF2(model, newton_tol=1e-11)
    dt = period / steps_per_tide
    y = interleave(np.full(grid.size, h0), np.full(grid.size, z0))
    y = stepper.run(y, 0.0, (cycles - 1) * period, dt)
    snaps = []
    stepper.run(y, (cycles - 1) * period, cycles * period, dt,
                observe=lambda t, s: snaps.append(s.copy()))
    snaps = np.array(snaps[1:])
    h_s, z_s = snaps[:, 0::2], snaps[:, 1::2]
    phase = 2.0 * np.pi * np.arange(1, steps_per_tide + 1) / steps_per_tide
    basis = np.exp(-1j * phase)[:, None]
    h_hat = 2.0 * np.mean(h_s * basis, axis=0)
    z_hat = 2.0 * np.mean(z_s * basis, axis=0)
    return {"x": grid.centers,
            "h_hat_re": h_hat.real, "h_hat_im": h_hat.imag,
            "z_hat_re": z_hat.real, "z_hat_im": z_hat.imag,
            "h_mean": h_s.mean(axis=0) - h0,
            "z_mean": z_s.mean(axis=0) - z0}


def uniform_tide(amplitude=0.2, period=M2_PERIOD_DAYS, fresh=10.0,
                 salt=10.0, length=60.0, cycles=2500, steps_per_tide=32,
                 aquifer=REFERENCE_AQUIFER):
    """Uniform layers in a closed domain driven by a periodic sea.

    Returns the fundamental harmonic and the period mean of the last
    cycle, for comparison with :mod:`linear`. The default length of the
    run (2500 cycles) lets the slowest baroclinic mode of the 60 m
    domain decay by more than four orders of magnitude.
    """
    return cached("uniform_tide", _uniform_tide, amplitude=amplitude,
                  period=period, fresh=fresh, salt=salt, length=length,
                  cycles=cycles, steps_per_tide=steps_per_tide,
                  aquifer=aquifer)


# -- wave setup ------------------------------------------------------------
def quasi_static_toe(aquifer, kind, sea_level, discharge, head, length):
    """Steady toe for a constant sea level (Strack-Glover lens)."""
    if kind == "head":
        discharge = core.discharge_for_head(aquifer, head, length,
                                            sea_level)
    return core.toe_length_flux(aquifer, discharge, sea_level)


def _setup_steady(kind, levels, aquifer, numerics, discharge):
    grid = make_grid(numerics)
    inland = make_inland(kind, aquifer, discharge, numerics.length)
    numeric, exact = [], []
    for level in levels:
        # Start Newton from the exact lens of this setup, whose discharge
        # differs from the reference one in a head-controlled aquifer.
        q_level = (core.discharge_for_head(aquifer, inland.head,
                                           numerics.length, level)
                   if kind == "head" else discharge)
        y = initial_lens(grid, aquifer, inland, q_level, level)
        numeric.append(core.toe_position(grid.centers, y[0::2], y[1::2],
                                         aquifer.delta, aquifer.base_depth))
        exact.append(quasi_static_toe(aquifer, kind, level, discharge,
                                      inland.head, numerics.length))
    return {"level": np.asarray(levels), "toe": np.array(numeric),
            "toe_exact": np.array(exact)}


def setup_steady(kind="flux", levels=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5),
                 aquifer=REFERENCE_AQUIFER, numerics=REFERENCE_NUMERICS,
                 discharge=REFERENCE_DISCHARGE):
    """Discrete steady toe for several constant setups.

    The head-controlled aquifer keeps the inland head of the reference
    lens, so its discharge drops as the setup rises.
    """
    return cached("setup_steady", _setup_steady, kind=kind,
                  levels=list(levels), aquifer=aquifer, numerics=numerics,
                  discharge=discharge)


def _setup_cycle(kind, period, amplitude, cycles, steps, aquifer, numerics,
                 discharge):
    grid = make_grid(numerics)
    inland = make_inland(kind, aquifer, discharge, numerics.length)
    y0 = initial_lens(grid, aquifer, inland, discharge)
    sea = SeaForcing(tide_amplitude=0.0, setup_amplitude=amplitude,
                     setup_period=period, ramp=0.25 * period)
    model = TwoLayerAquifer(grid, aquifer, _beach(sea), inland)
    stepper = TRBDF2(model, newton_tol=1e-10)
    d, b = aquifer.delta, aquifer.base_depth
    out = []
    stepper.run(y0, 0.0, cycles * period, period / steps,
                observe=lambda t, y: out.append(
                    (t, core.toe_position(grid.centers, y[0::2], y[1::2],
                                          d, b))))
    out = np.array(out)
    last = out[-steps - 1:-1]
    phase = 2.0 * np.pi * last[:, 0] / period
    response = 2.0 * np.mean(last[:, 1] * np.exp(-1j * phase))
    # The setup is -amplitude cos(phase), so divide by -amplitude.
    gain = response / (-amplitude)
    return {"period": period, "gain_re": gain.real, "gain_im": gain.imag,
            "t": out[:, 0], "toe": out[:, 1]}


def setup_cycle(kind="flux", period=365.25, amplitude=0.05, cycles=8,
                steps=200, aquifer=REFERENCE_AQUIFER,
                numerics=REFERENCE_NUMERICS, discharge=REFERENCE_DISCHARGE):
    """Toe response to a sinusoidal setup, in meters of toe per meter.

    ``gain`` is complex: its modulus is the toe excursion per meter of
    setup amplitude and its argument the phase lead of the toe.
    """
    return cached("setup_cycle", _setup_cycle, kind=kind, period=period,
                  amplitude=amplitude, cycles=cycles, steps=steps,
                  aquifer=aquifer, numerics=numerics, discharge=discharge)


def quasi_static_gain(aquifer, kind, discharge, head, length, eps=1e-4):
    """Slope of the steady toe with respect to mean sea level (m/m)."""
    up = quasi_static_toe(aquifer, kind, eps, discharge, head, length)
    down = quasi_static_toe(aquifer, kind, -eps, discharge, head, length)
    return (up - down) / (2.0 * eps)


def _seasonal(kind, tide_amplitude, setup_mean, setup_amplitude, years,
              steps_per_tide, frame_days, aquifer, numerics, discharge):
    grid = make_grid(numerics)
    inland = make_inland(kind, aquifer, discharge, numerics.length)
    y0 = initial_lens(grid, aquifer, inland, discharge)
    sea = SeaForcing(tide_amplitude=tide_amplitude, setup_mean=setup_mean,
                     setup_amplitude=setup_amplitude, ramp=30.0)
    model = TwoLayerAquifer(grid, aquifer, _beach(sea), inland)
    stepper = TRBDF2(model, newton_tol=1e-10)
    d, b = aquifer.delta, aquifer.base_depth
    x = grid.centers
    if tide_amplitude > 0.0:
        period = sea.tide_period
        dt = period / steps_per_tide
        window = steps_per_tide
    else:
        period = 1.0
        dt = 0.25
        window = 4
    total = years * 365.25
    steps = int(round(total / dt))
    frame_every = max(int(round(frame_days / (window * dt))), 1)

    acc = np.zeros(2 * grid.size)
    count = [0]
    series, frames, frame_t = [], [], []

    def observe(t, y):
        k = int(round(t / dt))
        if k == 0:
            return
        acc[:] += y
        count[0] += 1
        if k % window == 0:
            mean = acc / count[0]
            t_mid = t - 0.5 * window * dt
            series.append((t_mid, core.toe_position(
                x, mean[0::2], mean[1::2], d, b), model.volumes(mean)[1],
                float(sea.start_factor(t_mid) * sea.setup(t_mid))))
            if len(series) % frame_every == 0:
                frames.append(mean.copy())
                frame_t.append(t_mid)
            acc[:] = 0.0
            count[0] = 0

    stepper.run(y0, 0.0, steps * dt, dt, observe=observe)
    series = np.array(series)
    quasi = np.array([quasi_static_toe(aquifer, kind, s, discharge,
                                       inland.head, numerics.length)
                      for s in series[:, 3]])
    frames = np.array(frames)
    return {"t": series[:, 0], "toe": series[:, 1],
            "salt_volume": series[:, 2], "setup": series[:, 3],
            "toe_quasi_static": quasi, "x": x,
            "frame_t": np.array(frame_t), "frame_h": frames[:, 0::2],
            "frame_z": frames[:, 1::2], "toe0": core.toe_position(
                x, y0[0::2], y0[1::2], d, b)}


def seasonal(kind="head", tide_amplitude=0.5, setup_mean=0.2,
             setup_amplitude=0.15, years=4, steps_per_tide=32,
             frame_days=4.0, aquifer=REFERENCE_AQUIFER,
             numerics=REFERENCE_NUMERICS, discharge=REFERENCE_DISCHARGE):
    """Lens under tide plus an annual (monsoon-swell) setup cycle.

    Values are averaged over each tidal cycle (or over one day without
    a tide); frames of the averaged lens are kept every ``frame_days``.
    """
    return cached("seasonal", _seasonal, kind=kind,
                  tide_amplitude=tide_amplitude, setup_mean=setup_mean,
                  setup_amplitude=setup_amplitude, years=years,
                  steps_per_tide=steps_per_tide, frame_days=frame_days,
                  aquifer=aquifer, numerics=numerics, discharge=discharge)


# -- study design ----------------------------------------------------------
AMPLITUDES = (0.25, 0.5, 0.75, 1.0)
SETUP_PERIODS = (15.0, 30.0, 61.0, 122.0, 182.6, 365.25, 730.5, 1461.0,
                 2922.0, 5844.0)
UNIFORM_AMPLITUDES = (0.05, 0.1, 0.2, 0.4)
UNIFORM_PERIODS = (0.25, 1.0, 14.77, 60.0)


def sensitivity_cases():
    """Single-parameter departures from the reference tidal lens run.

    The base depth is changed together with the discharge,
    ``Q = 2 (B / 20)^2``, so that the toe stays 77 m from the shore.
    """
    a = REFERENCE_AQUIFER
    n = REFERENCE_NUMERICS
    return {
        "reference": {},
        "B = 10 m": {"aquifer": a.with_(base_depth=10.0), "discharge": 0.5},
        "B = 30 m": {"aquifer": a.with_(base_depth=30.0), "discharge": 4.5},
        "K = 15 m/d": {"aquifer": a.with_(conductivity=15.0),
                       "discharge": 1.0},
        "n = 0.2": {"aquifer": a.with_(porosity=0.2)},
        "diurnal tide": {"period": 1.0},
        "grid halved": {"numerics": Numerics(
            n.length, 0.5 * n.dx_min, 0.5 * n.dx_max,
            1.0 + 0.5 * (n.growth - 1.0))},
        "64 steps per tide": {"steps_per_tide": 64},
    }


def tidal_jobs():
    """Every tidal lens run used in the figures and reports."""
    jobs = [dict(amplitude=amp, kind=kind) for kind in ("flux", "head")
            for amp in AMPLITUDES]
    for name, change in sensitivity_cases().items():
        if name != "reference":
            jobs.append(dict(amplitude=0.5, kind="flux", **change))
    return jobs


def precompute(workers=None):
    """Run every cached experiment, the slow ones in parallel."""
    run_parallel(tidal_lens, tidal_jobs(), workers)
    jobs = [dict(amplitude=amp) for amp in UNIFORM_AMPLITUDES]
    jobs += [dict(amplitude=0.02, period=p, cycles=40)
             for p in UNIFORM_PERIODS]
    run_parallel(uniform_tide, jobs, workers)
    jobs = [dict(kind=kind, tide_amplitude=tide) for kind in ("flux", "head")
            for tide in (0.5, 0.0)]
    run_parallel(seasonal, jobs, workers)
    jobs = [dict(kind=kind, period=p) for kind in ("flux", "head")
            for p in SETUP_PERIODS]
    run_parallel(setup_cycle, jobs, workers)
    for kind in ("flux", "head"):
        setup_steady(kind)


# -- closure of the tidal boundary layer -----------------------------------
def lens_closure_table():
    """Effective boundary and toe shift of every reference tidal run.

    Returns a dict keyed by control type, each an array with columns
    amplitude, ``phi_e``, ``s`` and the tidal-mean toe shift.
    """
    rows = {}
    for kind in ("flux", "head"):
        vals = []
        for amp in AMPLITUDES:
            run = tidal_lens(amplitude=amp, kind=kind)
            vals.append((amp, float(run["phi_e"]), float(run["shift"]),
                         float(run["toe_mean"] - run["toe0"])))
        rows[kind] = np.array(vals)
    return rows


def closure_coefficients(rows=None, aquifer=REFERENCE_AQUIFER):
    """Fit ``phi_e = c_phi A^2 / (4B)`` and ``s = c_s A^2`` to the runs.

    Both are least-squares fits through the origin in ``A^2`` over all
    amplitudes and both control types.
    """
    rows = lens_closure_table() if rows is None else rows
    allrows = np.vstack(list(rows.values()))
    a2 = allrows[:, 0] ** 2
    c_phi = np.sum(a2 * allrows[:, 1]) / np.sum(a2 * a2) * (
        4.0 * aquifer.base_depth)
    c_s = np.sum(a2 * allrows[:, 2]) / np.sum(a2 * a2)
    return float(c_phi), float(c_s)


def closure_toe_shift(kind, amplitude, c_phi, c_s, use_shift=True,
                      aquifer=REFERENCE_AQUIFER,
                      length=REFERENCE_NUMERICS.length,
                      discharge=REFERENCE_DISCHARGE):
    """Tidal toe shift predicted by the effective-boundary closure."""
    inland = make_inland(kind, aquifer, discharge, length)
    phi_e = c_phi * amplitude ** 2 / (4.0 * aquifer.base_depth)
    shift = c_s * amplitude ** 2 if use_shift else 0.0
    toe = predicted_toe(aquifer, kind, phi_e, shift, discharge, inland.head,
                        length)
    base = predicted_toe(aquifer, kind, 0.0, 0.0, discharge, inland.head,
                         length)
    return toe - base


def setup_toe_shift(kind, setup, aquifer=REFERENCE_AQUIFER,
                    length=REFERENCE_NUMERICS.length,
                    discharge=REFERENCE_DISCHARGE):
    """Steady toe shift caused by a constant mean setup."""
    inland = make_inland(kind, aquifer, discharge, length)
    return (quasi_static_toe(aquifer, kind, setup, discharge, inland.head,
                             length)
            - quasi_static_toe(aquifer, kind, 0.0, discharge, inland.head,
                               length))
