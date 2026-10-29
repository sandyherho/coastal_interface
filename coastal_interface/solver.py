"""Finite-volume solver for the two-layer sharp-interface aquifer.

Governing equations
-------------------
Under the Dupuit assumption each layer is in vertical hydrostatic
equilibrium. With effective porosity n, the volume balances of the
freshwater layer (thickness ``h - zeta``) and of the saltwater layer
(thickness ``zeta + B``) are

    n d(h - zeta)/dt = d/dx [ K (h - zeta) dh/dx ]          + R,
    n d zeta / dt    = d/dx [ K (zeta + B) d(h + delta zeta)/dx ].

The first flux is the freshwater Darcy flux, the second the saltwater
Darcy flux written with the potential ``(1 + delta) phi = h + delta
zeta``; equal viscosities are assumed. Adding them gives the barotropic
balance ``n dh/dt = d/dx(total flux) + R``.

Discretization
--------------
Cell-centered finite volumes on a graded grid. Face fluxes use a
two-point gradient and an arithmetic-mean layer thickness, which makes
the steady freshwater flux the exact difference of the Kirchhoff
potential ``K (1 + delta) (h - eta)^2 / (2 delta)`` when the saltwater
is at rest. Where one cell has drained a layer (the lens pinch-out at
the shoreline, the saltwater toe), the face thickness is capped at
twice the thickness of the cell the flow comes from, so that a layer
cannot be pulled out of a cell that holds none of it. The cap is
inactive wherever neighboring thicknesses differ by less than a factor
of three, so the scheme stays second order in smooth regions.

Time stepping uses TR-BDF2 (trapezoidal stage followed by BDF2), which
is second-order accurate and L-stable; both stages share one Newton
matrix, factored in banded form.
"""

import numpy as np
from scipy.linalg.lapack import dgbtrf, dgbtrs

TRBDF2_GAMMA = 2.0 - np.sqrt(2.0)


class Grid:
    """Cell-centered grid on ``[0, length]`` graded away from the shore.

    Parameters
    ----------
    length : float
        Domain length (m).
    dx_min, dx_max : float
        Smallest (shoreline) and largest cell widths (m).
    growth : float
        Ratio of neighboring cell widths in the graded part.
    """

    def __init__(self, length, dx_min, dx_max=None, growth=1.0):
        dx_max = dx_min if dx_max is None else dx_max
        widths = []
        total = 0.0
        width = dx_min
        while total < length:
            widths.append(width)
            total += width
            width = min(width * growth, dx_max)
        widths = np.array(widths)
        widths *= length / widths.sum()
        self._build(np.concatenate(([0.0], np.cumsum(widths))))

    @classmethod
    def from_faces(cls, faces):
        """Build a grid from an explicit array of face positions."""
        grid = cls.__new__(cls)
        grid._build(np.asarray(faces, dtype=float))
        return grid

    def _build(self, faces):
        self.faces = faces
        self.centers = 0.5 * (faces[1:] + faces[:-1])
        self.widths = np.diff(faces)
        self.size = len(self.centers)
        # Distance between the two values that define each face gradient:
        # the boundary faces use the half cell next to them.
        distance = np.empty(self.size + 1)
        distance[1:-1] = np.diff(self.centers)
        distance[0] = self.centers[0] - faces[0]
        distance[-1] = faces[-1] - self.centers[-1]
        self.face_distance = distance

    @property
    def length(self):
        """Domain length (m)."""
        return float(self.faces[-1])


def _face_thickness(b_left, b_right, potential_left, potential_right):
    """Arithmetic-mean face thickness, capped by the upstream cell."""
    b_left = np.maximum(b_left, 0.0)
    b_right = np.maximum(b_right, 0.0)
    mean = 0.5 * (b_left + b_right)
    # Flow runs from high to low potential.
    donor = np.where(potential_left >= potential_right, b_left, b_right)
    return np.minimum(mean, 2.0 * donor)


class TwoLayerAquifer:
    """Semi-discrete two-layer aquifer, ``dy/dt = f(t, y)``.

    The state vector interleaves the two unknowns cell by cell,
    ``y = [h_0, zeta_0, h_1, zeta_1, ...]``, so that the Jacobian is
    banded with three diagonals on either side.

    Parameters
    ----------
    grid : Grid
        Finite-volume grid.
    aquifer : scenario.Aquifer
        Material properties.
    shore : callable
        ``shore(t) -> (h_b, zeta_b)``, the values imposed at ``x = 0``.
        For a beach face these are both the sea level, so that the
        freshwater lens pinches out at the shoreline.
    inland : scenario.Inland or similar
        Landward condition. ``kind="flux"`` imposes a seaward freshwater
        discharge, ``kind="head"`` a freshwater head; the saltwater
        layer is closed in both cases. ``kind="dirichlet"`` with a
        callable ``values(t) -> (h, zeta)`` fixes both unknowns and is
        used only for verification.
    source : callable, optional
        ``source(t, x) -> (s_fresh, s_salt)`` volumetric sources per unit
        area (m/day), used only for manufactured-solution tests.
    """

    def __init__(self, grid, aquifer, shore, inland, source=None):
        self.grid = grid
        self.aquifer = aquifer
        self.shore = shore
        self.inland = inland
        self.kind = inland.kind
        self.source = source
        self.size = 2 * grid.size
        # Constant factors of the discrete fluxes and divergences.
        self._k_over_dist = aquifer.conductivity / grid.face_distance
        self._inv_storage = 1.0 / (aquifer.porosity * grid.widths)
        n_ext = grid.size + 2
        self._h = np.empty(n_ext)
        self._z = np.empty(n_ext)

    # -- fluxes ----------------------------------------------------------
    def face_fluxes(self, t, y):
        """Freshwater and saltwater fluxes at all faces (m^2/day).

        Positive fluxes point landward (+x).
        """
        a = self.aquifer
        h_ext, z_ext = self._h, self._z

        # Extend each field by the shoreline value and an inland value;
        # the inland closure itself is applied below.
        h_ext[0], z_ext[0] = self.shore(t)
        h_ext[1:-1] = y[0::2]
        z_ext[1:-1] = y[1::2]
        h_ext[-1] = h_ext[-2]
        z_ext[-1] = z_ext[-2]
        if self.kind == "head":
            h_ext[-1] = self.inland.head
        elif self.kind == "dirichlet":
            h_ext[-1], z_ext[-1] = self.inland.values(t)

        bf = h_ext - z_ext
        bs = z_ext + a.base_depth
        pot_s = h_ext + a.delta * z_ext

        bf_face = _face_thickness(bf[:-1], bf[1:], h_ext[:-1], h_ext[1:])
        bs_face = _face_thickness(bs[:-1], bs[1:], pot_s[:-1], pot_s[1:])
        q_fresh = -self._k_over_dist * bf_face * (h_ext[1:] - h_ext[:-1])
        q_salt = -self._k_over_dist * bs_face * (pot_s[1:] - pot_s[:-1])

        if self.kind == "flux":
            # A seaward discharge Q is a flux of -Q in the +x direction.
            q_fresh[-1] = -self.inland.discharge
            q_salt[-1] = 0.0
        elif self.kind == "head":
            q_salt[-1] = 0.0
        return q_fresh, q_salt

    # -- right-hand side -------------------------------------------------
    def rhs(self, t, y):
        """Time derivative of the interleaved state."""
        a = self.aquifer
        q_fresh, q_salt = self.face_fluxes(t, y)

        fresh_rate = (q_fresh[:-1] - q_fresh[1:]) * self._inv_storage
        salt_rate = (q_salt[:-1] - q_salt[1:]) * self._inv_storage
        if a.recharge != 0.0:
            fresh_rate += a.recharge / a.porosity
        if self.source is not None:
            s_fresh, s_salt = self.source(t, self.grid.centers)
            fresh_rate += s_fresh / a.porosity
            salt_rate += s_salt / a.porosity

        dydt = np.empty_like(y)
        dydt[1::2] = salt_rate
        dydt[0::2] = salt_rate + fresh_rate
        return dydt

    # -- Jacobian --------------------------------------------------------
    def _coloring(self):
        """Index sets for the colored finite-difference Jacobian.

        Unknowns three cells apart never appear in the same equation, so
        the six unknowns of every block of three cells can be perturbed
        together. For each color this returns the perturbed columns and
        the (row, column) pairs of the band they fill.
        """
        if getattr(self, "_colors", None) is None:
            n = self.size
            colors = []
            for color in range(6):
                cols = np.arange(color, n, 6)
                cell = cols // 2
                # Rows of the owning cell and of its two neighbors.
                rows = 2 * cell[:, None] - 2 + np.arange(6)[None, :]
                owner = np.repeat(cols[:, None], 6, axis=1)
                keep = (rows >= 0) & (rows < n)
                colors.append((cols, rows[keep], owner[keep]))
            self._colors = colors
        return self._colors

    def banded_jacobian(self, t, y, f0=None):
        """Finite-difference Jacobian in LAPACK band storage.

        The returned array has three superdiagonals, the diagonal and
        three subdiagonals in rows 3 to 9, with rows 0 to 2 left free
        for the fill-in of the banded LU factorization.
        """
        f0 = self.rhs(t, y) if f0 is None else f0
        band = np.zeros((10, self.size))
        for cols, rows, owner in self._coloring():
            step = 1e-7 * np.maximum(1.0, np.abs(y[cols]))
            yp = y.copy()
            yp[cols] += step
            df = self.rhs(t, yp) - f0
            eps = np.zeros(self.size)
            eps[cols] = step
            band[6 + rows - owner, owner] = df[rows] / eps[owner]
        return band

    # -- budget ----------------------------------------------------------
    def volumes(self, y):
        """Freshwater and saltwater volumes per unit shoreline (m^2)."""
        a = self.aquifer
        h = y[0::2]
        zeta = y[1::2]
        w = self.grid.widths
        return (a.porosity * np.sum((h - zeta) * w),
                a.porosity * np.sum((zeta + a.base_depth) * w))


class TRBDF2:
    """TR-BDF2 time integrator with a banded Newton solver.

    Both implicit stages solve ``y - (gamma/2) dt f(t, y) = r``, so one
    banded LU factorization of ``I - (gamma/2) dt J`` serves the whole
    step. The Jacobian is evaluated once per step at the start of the
    step and refreshed inside a stage only if the simplified Newton
    iteration stalls.

    Parameters
    ----------
    model : TwoLayerAquifer
        Semi-discrete system.
    newton_tol : float
        Tolerance on the Newton update, relative to ``1 + max|y|``.
    max_newton : int
        Iterations allowed before the Newton matrix is refreshed.
    on_step : callable, optional
        ``on_step(dt, stages)`` called after every completed step with
        the three TR-BDF2 stages ``[(t, y), (t + gamma dt, y_gamma),
        (t + dt, y_new)]``; used for exact volume budgets.
    """

    def __init__(self, model, newton_tol=1e-11, max_newton=8,
                 on_step=None):
        self.model = model
        self.newton_tol = newton_tol
        self.max_newton = max_newton
        self.on_step = on_step
        self._lu = None
        self.jacobian_updates = 0
        self.newton_iterations = 0
        self.halvings = 0

    def _factor(self, t, y, dt, f0=None):
        coef = 0.5 * TRBDF2_GAMMA * dt
        band = -coef * self.model.banded_jacobian(t, y, f0)
        band[6] += 1.0
        lu, piv, info = dgbtrf(band, 3, 3)
        if info != 0:
            raise RuntimeError("singular Newton matrix")
        self._lu = (lu, piv)
        self.jacobian_updates += 1

    def _solve(self, rhs):
        lu, piv = self._lu
        x, info = dgbtrs(lu, 3, 3, rhs, piv)
        return x

    def _solve_stage(self, t, guesses, rest, dt):
        """Solve ``y - (gamma/2) dt f(t, y) = rest`` for ``y``.

        Simplified Newton iterations reuse the current factorization;
        when they contract slowly (ratio of successive updates above
        one half) the Newton matrix is refreshed at the current iterate.
        If an iterate stops being finite, the next guess is tried.
        """
        coef = 0.5 * TRBDF2_GAMMA * dt
        for attempt, guess in enumerate(guesses):
            if attempt > 0:
                self._factor(t, guess, dt)
            y = guess.copy()
            previous = np.inf
            refreshes = 0
            for _ in range(4 * self.max_newton):
                residual = y - coef * self.model.rhs(t, y) - rest
                update = self._solve(-residual)
                y += update
                self.newton_iterations += 1
                if not np.all(np.isfinite(y)):
                    break
                size = np.max(np.abs(update))
                if size < self.newton_tol * (1.0 + np.max(np.abs(y))):
                    return y
                if size > 0.5 * previous and refreshes < 3:
                    self._factor(t, y, dt)
                    refreshes += 1
                previous = size
        raise RuntimeError("Newton iteration did not converge")

    def step(self, t, y, dt, depth=0):
        """Advance by ``dt``, halving the step if Newton fails.

        Halving is a safeguard against abrupt forcing; the number of
        halvings is recorded so that runs can report it.
        """
        try:
            return self._step(t, y, dt)
        except RuntimeError:
            if depth >= 6:
                raise
            self.halvings += 1
            y_half = self.step(t, y, 0.5 * dt, depth + 1)
            return self.step(t + 0.5 * dt, y_half, 0.5 * dt, depth + 1)

    def _step(self, t, y, dt):
        """Advance the state by one TR-BDF2 step of length ``dt``."""
        g = TRBDF2_GAMMA
        f_n = self.model.rhs(t, y)
        self._factor(t, y, dt, f_n)
        # Trapezoidal stage to t + g dt.
        rest = y + 0.5 * g * dt * f_n
        euler = y + g * dt * f_n
        y_mid = self._solve_stage(t + g * dt, [euler, y], rest, dt)
        # BDF2 stage to t + dt; its implicit coefficient also equals g / 2.
        rest = (y_mid - (1.0 - g) ** 2 * y) / (g * (2.0 - g))
        extrapolated = y_mid + (1.0 - g) / g * (y_mid - y)
        y_new = self._solve_stage(t + dt, [extrapolated, y_mid], rest, dt)
        if self.on_step is not None:
            self.on_step(dt, [(t, y), (t + g * dt, y_mid), (t + dt, y_new)])
        return y_new

    def run(self, y0, t0, t1, dt, observe=None, every=1):
        """Integrate from ``t0`` to ``t1`` with a fixed step.

        Parameters
        ----------
        observe : callable, optional
            ``observe(t, y)`` called at the start and every ``every``
            steps.

        Returns
        -------
        ndarray
            State at ``t1``.
        """
        steps = int(round((t1 - t0) / dt))
        dt = (t1 - t0) / steps
        y = np.array(y0, dtype=float)
        if observe is not None:
            observe(t0, y)
        for k in range(1, steps + 1):
            y = self.step(t0 + (k - 1) * dt, y, dt)
            if observe is not None and k % every == 0:
                observe(t0 + k * dt, y)
        return y


class BoundaryBudget:
    """Volumes that cross the two boundaries during a TR-BDF2 run.

    Eliminating the stage value from the two TR-BDF2 stages shows that a
    step changes the state by ``dt [w f(t_n) + w f(t_gamma) + w_1
    f(t_n+1)]`` with ``w = 1 / (2 (2 - gamma))`` and ``w_1 = (1 - gamma) /
    (2 - gamma)``. Summing the boundary fluxes with the same weights
    gives the volumes that the discrete scheme actually exchanges, so the
    budget closes to the Newton tolerance.
    """

    def __init__(self, model):
        self.model = model
        g = TRBDF2_GAMMA
        self.weights = (0.5 / (2.0 - g), 0.5 / (2.0 - g),
                        (1.0 - g) / (2.0 - g))
        self.fresh_in = 0.0
        self.salt_in = 0.0
        self.recharge_in = 0.0

    def __call__(self, dt, stages):
        """Accumulate one step; positive values enter the aquifer."""
        for weight, (t, y) in zip(self.weights, stages):
            q_fresh, q_salt = self.model.face_fluxes(t, y)
            self.fresh_in += weight * dt * (q_fresh[0] - q_fresh[-1])
            self.salt_in += weight * dt * (q_salt[0] - q_salt[-1])
        self.recharge_in += dt * self.model.aquifer.recharge * (
            self.model.grid.length)


def steady_state(model, y0, t=0.0, tol=1e-12, max_iter=60):
    """Solve ``f(t, y) = 0`` by Newton's method with a banded Jacobian.

    A damped step (halving until the residual decreases) keeps the
    iteration inside the region where both layer thicknesses are
    positive.
    """
    y = np.array(y0, dtype=float)
    f = model.rhs(t, y)
    for _ in range(max_iter):
        band = model.banded_jacobian(t, y, f)
        lu, piv, info = dgbtrf(band, 3, 3)
        if info != 0:
            raise RuntimeError("singular Jacobian")
        update, info = dgbtrs(lu, 3, 3, -f, piv)
        norm = np.max(np.abs(f))
        scale = 1.0
        while scale > 1e-6:
            trial = y + scale * update
            f_trial = model.rhs(t, trial)
            if np.max(np.abs(f_trial)) < norm or scale < 1e-3:
                break
            scale *= 0.5
        y, f = trial, f_trial
        if np.max(np.abs(scale * update)) < tol * (1.0 + np.max(np.abs(y))):
            return y
    raise RuntimeError("steady Newton iteration did not converge")


def interleave(h, zeta):
    """Pack the two unknowns into the interleaved state vector."""
    y = np.empty(2 * len(h))
    y[0::2] = h
    y[1::2] = zeta
    return y


def split(y):
    """Unpack the interleaved state vector into ``(h, zeta)``."""
    return y[0::2].copy(), y[1::2].copy()
