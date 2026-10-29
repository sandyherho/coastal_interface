"""Linear and weakly nonlinear theory of a uniform two-layer aquifer.

Linearizing the two-layer equations about uniform layers of thickness
``H_f`` (fresh) and ``H_s`` (salt) at rest gives, for the perturbation
``u = (h', zeta')``,

    du/dt = D d2u/dx2,    D = (K / n) [[H_f + H_s, delta H_s],
                                       [H_s,       delta H_s]].

The trace and determinant of ``D`` are ``(K/n)(H + delta H_s)`` and
``(K/n)^2 delta H_f H_s`` with ``H = H_f + H_s``, so its eigenvalues are
real, positive and distinct:

    D_0, D_1 = T/2 +/- sqrt(T^2/4 - det).

``D_0`` is close to ``K H / n``, the classical diffusivity of the whole
saturated thickness; ``D_1`` is close to ``delta K H_f H_s / (n H)``,
the porous-medium counterpart of the reduced-gravity combination
``g' H_1 H_2 / (H_1 + H_2)`` of two-layer flow. Their ratio never
exceeds ``delta / 4`` to leading order.

The mode amplitudes obey uncoupled diffusion equations. A periodic
sea level ``A cos(omega t)`` imposed on both unknowns at ``x = 0``
(the whole column rises and falls with the sea at the shoreline)
excites each mode with a complex decay rate
``kappa_j = (1 + i) / L_j``, ``L_j = sqrt(2 D_j / omega)``.

Averaging the nonlinear fluxes over a period gives the rectified
(time-mean) heads at second order in the amplitude.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Layers:
    """Uniform two-layer state about which the theory is built."""

    conductivity: float
    porosity: float
    delta: float
    fresh: float
    salt: float

    @classmethod
    def from_aquifer(cls, aquifer, fresh, salt):
        """Build from a :class:`scenario.Aquifer` and two thicknesses."""
        return cls(aquifer.conductivity, aquifer.porosity, aquifer.delta,
                   fresh, salt)

    @property
    def total(self):
        """Total saturated thickness H (m)."""
        return self.fresh + self.salt


def diffusion_matrix(layers):
    """Matrix ``D`` of the linear system for ``(h', zeta')`` (m^2/day)."""
    c = layers.conductivity / layers.porosity
    hf, hs, d = layers.fresh, layers.salt, layers.delta
    return c * np.array([[hf + hs, d * hs],
                         [hs, d * hs]])


def mode_diffusivities(layers):
    """Exact barotropic and baroclinic diffusivities ``(D_0, D_1)``."""
    c = layers.conductivity / layers.porosity
    trace = c * (layers.total + layers.delta * layers.salt)
    det = c ** 2 * layers.delta * layers.fresh * layers.salt
    root = np.sqrt(0.25 * trace ** 2 - det)
    d0 = 0.5 * trace + root
    # Written as det / d0 to avoid cancellation for small delta.
    return d0, det / d0


def leading_order_diffusivities(layers):
    """Small-``delta`` forms ``K H / n`` and ``delta K H_f H_s / (n H)``."""
    c = layers.conductivity / layers.porosity
    return (c * layers.total,
            c * layers.delta * layers.fresh * layers.salt / layers.total)


def modes(layers):
    """Diffusivities and eigenvectors, barotropic mode first.

    Eigenvectors are scaled so that their head component is one for
    the barotropic mode and their interface component is one for the
    baroclinic mode.
    """
    d0, d1 = mode_diffusivities(layers)
    c = layers.conductivity / layers.porosity
    hs, d = layers.salt, layers.delta
    # Second row of (D - lambda I) v = 0: c H_s v_h + (c d H_s - lambda) v_z.
    v0 = np.array([1.0, c * hs / (d0 - c * d * hs)])
    v1 = np.array([(d1 - c * d * hs) / (c * hs), 1.0])
    return np.array([d0, d1]), np.column_stack([v0, v1])


def skin_depths(layers, omega):
    """Decay lengths ``L_j = sqrt(2 D_j / omega)`` of the two modes (m)."""
    return np.sqrt(2.0 * np.array(mode_diffusivities(layers)) / omega)


def periodic_response(layers, omega, x, amplitude=1.0, length=None):
    """Complex amplitudes of ``h'`` and ``zeta'`` for a periodic sea.

    Parameters
    ----------
    omega : float
        Angular frequency (rad/day).
    x : array_like
        Distances from the shoreline (m).
    amplitude : float
        Sea-level amplitude A, imposed on both unknowns at ``x = 0``.
    length : float, optional
        Position of a closed (no-flow) landward boundary; semi-infinite
        aquifer if omitted.

    Returns
    -------
    h_hat, zeta_hat : ndarray of complex
        ``h'(x, t) = Re[h_hat(x) exp(i omega t)]`` and likewise for the
        interface.
    """
    x = np.asarray(x, dtype=float)
    diff, vec = modes(layers)
    weights = np.linalg.solve(vec, np.array([amplitude, amplitude]))
    kappa = np.sqrt(1j * omega / diff)
    profiles = []
    for k in kappa:
        if length is None:
            profiles.append(np.exp(-k * x))
        else:
            # cosh(k (L - x)) / cosh(k L), written to avoid overflow.
            num = np.exp(-k * x) + np.exp(-k * (2.0 * length - x))
            profiles.append(num / (1.0 + np.exp(-2.0 * k * length)))
    profiles = np.array(profiles)
    h_hat = vec[0, 0] * weights[0] * profiles[0] + (
        vec[0, 1] * weights[1] * profiles[1])
    z_hat = vec[1, 0] * weights[0] * profiles[0] + (
        vec[1, 1] * weights[1] * profiles[1])
    return h_hat, z_hat


def _derivative(layers, omega, x, amplitude, length):
    """Exact x-derivatives of the complex amplitudes."""
    x = np.asarray(x, dtype=float)
    diff, vec = modes(layers)
    weights = np.linalg.solve(vec, np.array([amplitude, amplitude]))
    kappa = np.sqrt(1j * omega / diff)
    slopes = []
    for k in kappa:
        if length is None:
            slopes.append(-k * np.exp(-k * x))
        else:
            num = -np.exp(-k * x) + np.exp(-k * (2.0 * length - x))
            slopes.append(k * num / (1.0 + np.exp(-2.0 * k * length)))
    slopes = np.array(slopes)
    hx = vec[0, 0] * weights[0] * slopes[0] + vec[0, 1] * weights[1] * (
        slopes[1])
    zx = vec[1, 0] * weights[0] * slopes[0] + vec[1, 1] * weights[1] * (
        slopes[1])
    return hx, zx


def interface_transfer(layers, omega, x):
    """Interface response per unit sea-level amplitude, ``zeta_hat / A``."""
    return periodic_response(layers, omega, x)[1]


def tidal_fluxes(layers, omega, x, amplitude, length=None):
    """Period-averaged nonlinear fluxes at second order (m^2/day).

    Returns the mean freshwater and saltwater fluxes carried by the
    correlation of thickness and gradient fluctuations,

        F_f = -K < (h' - zeta') dh'/dx >,
        F_s = -K < zeta' d(h' + delta zeta')/dx >,

    the porous-medium analogue of a Stokes drift. Positive values point
    landward.
    """
    k, d = layers.conductivity, layers.delta
    h, z = periodic_response(layers, omega, x, amplitude, length)
    hx, zx = _derivative(layers, omega, x, amplitude, length)
    f_fresh = -0.5 * k * np.real((h - z) * np.conj(hx))
    f_salt = -0.5 * k * np.real(z * np.conj(hx + d * zx))
    return f_fresh, f_salt


def rectified_means(layers, omega, amplitude):
    """Far-field mean head and interface shifts in a closed aquifer.

    With no mean through-flow, each layer's mean Darcy flux must cancel
    its tidal flux,

        K H_f d<h>/dx            = F_f,
        K H_s d<h + delta zeta>/dx = F_s,

    so the far-field shifts follow from integrals of the exponentially
    decaying tidal fluxes, which are evaluated in closed form for a
    semi-infinite aquifer. The interface shift carries a factor
    ``1/delta``: the Ghyben-Herzberg lever acts on the rectified part of
    the response, not on the oscillation itself.

    Returns
    -------
    dict
        ``head`` and ``interface`` shifts (m), and the classical
        freshwater-only overheight ``A^2 / (4 H_f)`` for comparison.
    """
    k, d = layers.conductivity, layers.delta
    diff, vec = modes(layers)
    weights = np.linalg.solve(vec, np.array([amplitude, amplitude]))
    kappa = np.sqrt(1j * omega / diff)
    a_h = vec[0] * weights
    a_z = vec[1] * weights

    def mean_product(p, q):
        # Integral over (0, inf) of < p dq/dx > for fields that are sums
        # of exp(-kappa_j x) with mode coefficients p_j and q_j.
        total = 0.0
        for i in range(2):
            for j in range(2):
                total += p[i] * np.conj(-kappa[j] * q[j]) / (
                    kappa[i] + np.conj(kappa[j]))
        return 0.5 * np.real(total)

    int_fresh = -k * mean_product(a_h - a_z, a_h)
    int_salt = -k * mean_product(a_z, a_h + d * a_z)
    head = int_fresh / (k * layers.fresh)
    salt_potential = int_salt / (k * layers.salt)
    interface = (salt_potential - head) / d
    return {"head": head, "interface": interface,
            "salt_potential": salt_potential,
            "classical_overheight": amplitude ** 2 / (4.0 * layers.fresh)}


def rectified_profiles(layers, omega, amplitude, x, length=None):
    """Mean head and interface shifts as functions of ``x``.

    Numerical integration of the exact tidal fluxes; used to compare
    with the finite-volume solver in a closed domain.
    """
    x = np.asarray(x, dtype=float)
    fine = np.linspace(0.0, x.max(), 40001)
    f_f, f_s = tidal_fluxes(layers, omega, fine, amplitude, length)
    k = layers.conductivity
    grad_h = f_f / (k * layers.fresh)
    grad_p = f_s / (k * layers.salt)
    head = _cumulative_trapezoid(grad_h, fine)
    pot = _cumulative_trapezoid(grad_p, fine)
    head = np.interp(x, fine, head)
    pot = np.interp(x, fine, pot)
    return head, (pot - head) / layers.delta


def _cumulative_trapezoid(y, x):
    """Cumulative trapezoidal integral.

    On the 40001-point grids used here the relative error is of order
    ``(dx / L_1)^2 / 12``, below 1e-5 for every case reported.
    """
    out = np.zeros_like(y)
    out[1:] = np.cumsum(0.5 * (y[1:] + y[:-1]) * np.diff(x))
    return out


def rectified_closed_form(layers, amplitude):
    """Far-field rectified means in closed form.

    For fields that are sums of the two modes with real modal
    amplitudes ``f_j`` and ``g_j``, the period-averaged integral is

        int_0^inf < f dg/dx > dx = -f(0) g(0) / 4 - rho W(f, g) / 4,

    with ``rho = (D_0 - D_1) / (D_0 + D_1)`` and the modal Wronskian
    ``W(f, g) = f_0 g_1 - f_1 g_0``; the frequency drops out. Applied to
    the two flux balances, with ``W(h, zeta) = K A^2 H_f / (n (D_0 -
    D_1))``, it gives

        <h>    = A^2 / [4 (H + delta H_s)],
        <zeta> = <h> + A^2 / (4 H_s).

    :func:`rectified_means` evaluates the same integrals term by term
    and agrees to round-off.
    """
    total = layers.total + layers.delta * layers.salt
    head = amplitude ** 2 / (4.0 * total)
    return {"head": head,
            "interface": head + amplitude ** 2 / (4.0 * layers.salt)}
