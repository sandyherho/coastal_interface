"""Manufactured solutions for verifying the finite-volume solver.

A smooth pair ``(h_m, zeta_m)`` is chosen, and the source terms that
make it an exact solution of the two-layer equations are computed
analytically from its derivatives:

    s_f = n d(h - zeta)/dt - d/dx[ K (h - zeta) dh/dx ],
    s_s = n d zeta/dt      - d/dx[ K (zeta + B) dp/dx ],   p = h + delta zeta.

The fields keep both layers thick everywhere, so the thickness cap in
the face fluxes stays inactive and the test measures the formal order
of the scheme.
"""

from types import SimpleNamespace

import numpy as np


class Manufactured:
    """Smooth exact solution on ``[0, length]``.

    Parameters
    ----------
    aquifer : scenario.Aquifer
        Material properties.
    length : float
        Domain length (m).
    omega : float
        Angular frequency of the time dependence (rad/day); zero gives a
        steady solution.
    """

    def __init__(self, aquifer, length=50.0, omega=0.0):
        self.aquifer = aquifer
        self.length = length
        self.omega = omega
        self.k = np.pi / length

    # Fields and their derivatives. Each method returns a tuple
    # (value, d/dx, d2/dx2, d/dt).
    def _h(self, x, t):
        k, w = self.k, self.omega
        c, dc = np.cos(w * t), -w * np.sin(w * t)
        s, ds, dds = (np.sin(k * x), k * np.cos(k * x),
                      -k * k * np.sin(k * x))
        a, b = 0.6, 0.004
        value = 0.4 + a * s * (1.0 + 0.5 * c) + b * x
        return (value, a * ds * (1.0 + 0.5 * c) + b,
                a * dds * (1.0 + 0.5 * c), a * s * 0.5 * dc)

    def _zeta(self, x, t):
        k, w = self.k, self.omega
        s, ds = np.sin(w * t), w * np.cos(w * t)
        c, dc, ddc = (np.cos(1.5 * k * x), -1.5 * k * np.sin(1.5 * k * x),
                      -2.25 * k * k * np.cos(1.5 * k * x))
        a = 3.0
        value = -10.0 + a * c * (1.0 + 0.3 * s)
        return (value, a * dc * (1.0 + 0.3 * s), a * ddc * (1.0 + 0.3 * s),
                a * c * 0.3 * ds)

    def exact(self, x, t):
        """Exact water table and interface."""
        return self._h(x, t)[0], self._zeta(x, t)[0]

    def source(self, t, x):
        """Fresh and salt source terms (m/day)."""
        a = self.aquifer
        h, hx, hxx, ht = self._h(x, t)
        z, zx, zxx, zt = self._zeta(x, t)
        kk, n, d = a.conductivity, a.porosity, a.delta
        s_fresh = n * (ht - zt) - kk * ((hx - zx) * hx + (h - z) * hxx)
        px, pxx = hx + d * zx, hxx + d * zxx
        s_salt = n * zt - kk * (zx * px + (z + a.base_depth) * pxx)
        return s_fresh, s_salt

    def boundaries(self):
        """Shoreline callable and inland Dirichlet condition."""
        def shore(t):
            h, z = self.exact(0.0, t)
            return float(h), float(z)

        def inland_values(t):
            h, z = self.exact(self.length, t)
            return float(h), float(z)

        return shore, SimpleNamespace(kind="dirichlet", values=inland_values)

    def cell_average(self, faces, t, points=4):
        """Cell averages of the exact fields by Gauss-Legendre quadrature."""
        nodes, weights = np.polynomial.legendre.leggauss(points)
        left, right = faces[:-1], faces[1:]
        half = 0.5 * (right - left)
        mid = 0.5 * (right + left)
        h_avg = np.zeros_like(mid)
        z_avg = np.zeros_like(mid)
        for xi, wi in zip(nodes, weights):
            h, z = self.exact(mid + half * xi, t)
            h_avg += 0.5 * wi * h
            z_avg += 0.5 * wi * z
        return h_avg, z_avg
