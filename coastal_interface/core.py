"""Hydrostatics of the sharp interface and the steady Strack-Glover lens.

Notation
--------
h      freshwater head, equal to the water-table elevation under the
       Dupuit assumption (m above mean sea level)
zeta   elevation of the fresh-salt interface (m, negative below datum)
phi    saltwater head, defined so that the saltwater pressure is
       rho_s g (phi - z)
B      depth of the aquifer base below datum
delta  (rho_s - rho_f) / rho_f

Continuity of pressure across the interface,
``rho_f g (h - zeta) = rho_s g (phi - zeta)``, gives

    zeta = ((1 + delta) phi - h) / delta,

and the saltwater driving potential ``(1 + delta) phi = h + delta zeta``
that appears in the saltwater Darcy flux. When the saltwater is at rest
and its head equals the sea level ``eta``, the relation reduces to the
Ghyben-Herzberg rule ``eta - zeta = (h - eta) / delta``.
"""

import numpy as np


def interface_elevation(h, phi, delta):
    """Interface elevation from the fresh and salt heads."""
    return ((1.0 + delta) * phi - h) / delta


def saltwater_potential(h, zeta, delta):
    """Return ``(1 + delta) phi = h + delta zeta``.

    This is the saltwater head expressed as a freshwater-equivalent
    potential; its gradient drives the saltwater Darcy flux.
    """
    return h + delta * zeta


def ghyben_herzberg(h, sea_level, delta, base_depth):
    """Quasi-static Ghyben-Herzberg interface, clipped to the base.

    The saltwater is assumed to be at rest with its head equal to the
    instantaneous sea level. This is the reference against which the
    dynamic interface is compared.
    """
    zeta = sea_level - (h - sea_level) / delta
    return np.clip(zeta, -base_depth, None)


def fresh_thickness(h, zeta):
    """Thickness of the freshwater layer."""
    return h - zeta


def salt_thickness(zeta, base_depth):
    """Thickness of the saltwater layer above the base."""
    return zeta + base_depth


def toe_length_flux(aquifer, discharge, sea_level=0.0):
    """Toe distance of the steady lens for a fixed seaward discharge.

    ``x_T = K (1 + delta) delta (B + eta)^2 / (2 Q)``.
    """
    k, d, b = aquifer.conductivity, aquifer.delta, aquifer.base_depth
    return k * (1.0 + d) * d * (b + sea_level) ** 2 / (2.0 * discharge)


def discharge_for_head(aquifer, head, length, sea_level=0.0):
    """Seaward discharge of the steady lens for a fixed inland head.

    With the toe inside the domain the discharge is linear in the
    squared heads,

        Q = K [ (h_L + B)^2 - (1 + delta)(B + eta)^2 ] / (2 L),

    and when the interface extends past ``x = L`` it is

        Q = K (1 + delta) (h_L - eta)^2 / (2 delta L).
    """
    k, d, b = aquifer.conductivity, aquifer.delta, aquifer.base_depth
    q_toe = k * ((head + b) ** 2 - (1.0 + d) * (b + sea_level) ** 2) / (
        2.0 * length)
    q_open = k * (1.0 + d) * (head - sea_level) ** 2 / (2.0 * d * length)
    x_toe = toe_length_flux(aquifer, q_toe, sea_level) if q_toe > 0 else (
        np.inf)
    return q_toe if x_toe <= length else q_open


def steady_lens(aquifer, discharge, x, sea_level=0.0):
    """Steady water table and interface of the Strack-Glover lens.

    No recharge between the shoreline and ``x``; the lens pinches out
    at the shoreline, where ``h = zeta = eta``.

    Returns
    -------
    h, zeta : ndarray
        Water table and interface elevation at ``x``.
    """
    k, d, b = aquifer.conductivity, aquifer.delta, aquifer.base_depth
    x = np.asarray(x, dtype=float)
    x_toe = toe_length_flux(aquifer, discharge, sea_level)

    # Interface zone: (h - eta)^2 = 2 delta Q x / (K (1 + delta)).
    rise = np.sqrt(2.0 * d * discharge * np.minimum(x, x_toe)
                   / (k * (1.0 + d)))
    h = sea_level + rise
    zeta = sea_level - rise / d

    # Freshwater-only zone: (h + B)^2 grows linearly beyond the toe.
    inland = x > x_toe
    h_toe_total = (1.0 + d) * (b + sea_level)
    h[inland] = np.sqrt(h_toe_total ** 2 + 2.0 * discharge
                        * (x[inland] - x_toe) / k) - b
    zeta[inland] = -b
    return h, zeta


def toe_position(x_centers, h, zeta, delta, base_depth, margin=0.02):
    """Landward end of the saltwater wedge.

    In the interface zone of a lens whose saltwater is at rest, the
    Kirchhoff variable ``u = (h - phi)^2`` is linear in ``x`` and the toe
    is where ``u`` reaches ``(delta (B + phi))^2``. The toe is placed by
    extrapolating ``u`` linearly from the last two cells whose saltwater
    layer is thicker than ``margin * B``; the cell that contains the toe
    itself is skipped because its thickness carries the first-order
    error of the degenerate flux there. For the steady Strack-Glover
    lens the estimate is exact.
    """
    thick = np.nonzero(zeta + base_depth > margin * base_depth)[0]
    if thick.size < 2:
        return float(x_centers[0])
    i = thick[-1]
    if i == len(x_centers) - 1:
        return float(x_centers[-1])
    phi = (h + delta * zeta) / (1.0 + delta)
    u = (h - phi) ** 2
    u_toe = (delta * (base_depth + phi[i])) ** 2
    slope = (u[i] - u[i - 1]) / (x_centers[i] - x_centers[i - 1])
    if slope <= 0.0:
        return float(x_centers[i])
    return float(x_centers[i] + (u_toe - u[i]) / slope)
