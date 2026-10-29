"""Convergence tests of the solver with manufactured solutions."""

import numpy as np

from .io_utils import cached
from .manufactured import Manufactured
from .scenario import REFERENCE_AQUIFER
from .solver import TRBDF2, Grid, TwoLayerAquifer, interleave, steady_state


def _refine(faces, levels):
    """Split every cell in two, ``levels`` times."""
    for _ in range(levels):
        mid = 0.5 * (faces[1:] + faces[:-1])
        faces = np.sort(np.concatenate([faces, mid]))
    return faces


def _space(aquifer, levels):
    mms = Manufactured(aquifer, 50.0, 0.0)
    shore, inland = mms.boundaries()
    base = Grid(50.0, 0.25, 2.0, 1.1).faces
    cells, err_h, err_z = [], [], []
    for level in range(levels):
        grid = Grid.from_faces(_refine(base, level))
        model = TwoLayerAquifer(grid, aquifer, shore, inland,
                                source=mms.source)
        h, z = mms.exact(grid.centers, 0.0)
        y = steady_state(model, interleave(h + 0.01, z - 0.05))
        cells.append(grid.size)
        err_h.append(np.max(np.abs(y[0::2] - h)))
        err_z.append(np.max(np.abs(y[1::2] - z)))
    return {"cells": np.array(cells), "err_h": np.array(err_h),
            "err_z": np.array(err_z)}


def space_convergence(aquifer=REFERENCE_AQUIFER, levels=5):
    """Steady manufactured solution on a graded grid, refined by halving.

    Errors are the maximum differences between the cell values and the
    exact fields at the cell centers.
    """
    return cached("mms_space", _space, aquifer=aquifer, levels=levels)


def _time(aquifer, steps):
    mms = Manufactured(aquifer, 50.0, 2.0 * np.pi)
    shore, inland = mms.boundaries()
    grid = Grid(50.0, 0.25, 2.0, 1.1)
    model = TwoLayerAquifer(grid, aquifer, shore, inland, source=mms.source)
    h0, z0 = mms.exact(grid.centers, 0.0)
    finest = steps[-1] * 16
    solutions = {}
    for n in list(steps) + [finest]:
        stepper = TRBDF2(model, newton_tol=1e-13)
        solutions[n] = stepper.run(interleave(h0, z0), 0.0, 1.0, 1.0 / n)
    reference = solutions[finest]
    err = [np.max(np.abs(solutions[n] - reference)) for n in steps]
    return {"steps": np.array(steps), "err": np.array(err)}


def time_convergence(aquifer=REFERENCE_AQUIFER,
                     steps=(25, 50, 100, 200, 400)):
    """Time-periodic manufactured solution over one day.

    The error of each step size is measured against a run with sixteen
    times more steps on the same grid, so the spatial error cancels.
    """
    return cached("mms_time", _time, aquifer=aquifer, steps=list(steps))


def observed_order(sizes, errors):
    """Orders between successive refinements, ``log2(e_k / e_k+1)``."""
    ratio = np.asarray(sizes[1:], float) / np.asarray(sizes[:-1], float)
    return np.log(np.asarray(errors[:-1]) / np.asarray(errors[1:])) / (
        np.log(ratio))
