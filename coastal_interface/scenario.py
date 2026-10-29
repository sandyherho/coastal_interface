"""Parameters of the idealized coastal aquifer and its sea-side forcing.

Units are meters and days throughout, the units in which hydraulic
conductivity and recharge are usually quoted in hydrogeology. Angular
frequencies are therefore in rad/day and diffusivities in m^2/day.

The reference values describe a homogeneous, unconfined, clean-sand
aquifer on a tropical coast. They are illustrative choices within the
ranges given in standard hydrogeology texts, not measurements of any
particular site.
"""

from dataclasses import dataclass, replace

import numpy as np

# Principal lunar semidiurnal tide, 12.4206 h.
M2_PERIOD_DAYS = 12.4206012 / 24.0
YEAR_DAYS = 365.25


@dataclass(frozen=True)
class Aquifer:
    """Homogeneous unconfined aquifer with a sharp fresh-salt interface.

    Attributes
    ----------
    conductivity : float
        Freshwater hydraulic conductivity K (m/day).
    porosity : float
        Effective porosity n, taken equal to the specific yield.
    delta : float
        Density contrast (rho_s - rho_f) / rho_f.
    base_depth : float
        Depth B of the impermeable base below mean sea level (m).
    recharge : float
        Areal recharge R on the land surface (m/day).
    """

    conductivity: float = 30.0
    porosity: float = 0.30
    delta: float = 0.025
    base_depth: float = 20.0
    recharge: float = 0.0

    def with_(self, **changes):
        """Return a copy with some attributes changed."""
        return replace(self, **changes)


@dataclass(frozen=True)
class SeaForcing:
    """Sea level at the shoreline: mean, wave setup and tide.

    The shoreline sea level is
    ``eta(t) = mean_level + setup(t) + tide_amplitude * cos(omega t)``,
    where the setup is either constant or follows an annual cycle that
    stands in for a monsoon swell season.

    Attributes
    ----------
    tide_amplitude : float
        Amplitude A of a single tidal constituent (m).
    tide_period : float
        Tidal period (days).
    setup_mean : float
        Time-mean wave setup at the shoreline (m).
    setup_amplitude : float
        Amplitude of the annual setup cycle (m).
    setup_period : float
        Period of the setup cycle (days).
    mean_level : float
        Still-water mean sea level (m); zero defines the datum.
    ramp : float
        Time scale (days) of a smooth start, ``1 - exp(-(t/ramp)^2)``,
        applied to the tide and setup so that the aquifer is not hit by
        a sudden jump in sea level; zero switches it off.
    """

    tide_amplitude: float = 0.5
    tide_period: float = M2_PERIOD_DAYS
    setup_mean: float = 0.0
    setup_amplitude: float = 0.0
    setup_period: float = YEAR_DAYS
    mean_level: float = 0.0
    ramp: float = 0.0

    @property
    def tide_frequency(self):
        """Angular tidal frequency (rad/day)."""
        return 2.0 * np.pi / self.tide_period

    def setup(self, t):
        """Wave setup at time ``t`` (m)."""
        phase = 2.0 * np.pi * np.asarray(t) / self.setup_period
        return self.setup_mean - self.setup_amplitude * np.cos(phase)

    def start_factor(self, t):
        """Smooth start-up factor rising from 0 to 1."""
        if self.ramp <= 0.0:
            return 1.0
        return 1.0 - np.exp(-(np.asarray(t) / self.ramp) ** 2)

    def tide(self, t):
        """Tidal part of the sea level at time ``t`` (m)."""
        return (self.start_factor(t) * self.tide_amplitude
                * np.cos(self.tide_frequency * np.asarray(t)))

    def level(self, t):
        """Total shoreline sea level at time ``t`` (m)."""
        return (self.mean_level + self.start_factor(t) * self.setup(t)
                + self.tide(t))

    def with_(self, **changes):
        """Return a copy with some attributes changed."""
        return replace(self, **changes)


@dataclass(frozen=True)
class Inland:
    """Landward boundary condition of the freshwater layer.

    Attributes
    ----------
    kind : str
        ``"flux"`` fixes the seaward freshwater discharge per unit
        shoreline length; ``"head"`` fixes the freshwater head.
    discharge : float
        Seaward freshwater discharge Q (m^2/day) for ``"flux"``.
    head : float
        Freshwater head above mean sea level (m) for ``"head"``.
    """

    kind: str = "flux"
    discharge: float = 2.0
    head: float = 0.0


@dataclass(frozen=True)
class Numerics:
    """Grid and time-step settings.

    The grid is graded geometrically from ``dx_min`` at the shoreline,
    where the tidal boundary layers sit, up to ``dx_max``.
    """

    length: float = 300.0
    dx_min: float = 0.05
    dx_max: float = 1.0
    growth: float = 1.04


REFERENCE_AQUIFER = Aquifer()
REFERENCE_SEA = SeaForcing()
REFERENCE_INLAND = Inland()
REFERENCE_NUMERICS = Numerics()
