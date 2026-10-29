"""Output paths, run caching, CSV tables and plain-text reports."""

import hashlib
import json
import os
from dataclasses import asdict, is_dataclass

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUTS = os.path.join(ROOT, "outputs")
FIGURES = os.path.join(OUTPUTS, "figures")
ANIMATIONS = os.path.join(OUTPUTS, "animations")
DATA = os.path.join(OUTPUTS, "data")
REPORTS = os.path.join(OUTPUTS, "reports")
CACHE = os.path.join(OUTPUTS, "cache")


def ensure_dirs():
    """Create the output folders if they do not exist."""
    for path in (FIGURES, ANIMATIONS, DATA, REPORTS, CACHE):
        os.makedirs(path, exist_ok=True)


def _plain(value):
    """Turn parameters into JSON-friendly values for hashing."""
    if is_dataclass(value):
        return {k: _plain(v) for k, v in asdict(value).items()}
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def cached(name, function, **params):
    """Run ``function(**params)`` once and keep its result on disk.

    The result must be a dict of arrays and numbers. The cache key is a
    hash of the parameters, so changing any of them triggers a rerun.
    """
    ensure_dirs()
    key = hashlib.sha1(json.dumps(_plain(params), sort_keys=True)
                       .encode()).hexdigest()[:12]
    path = os.path.join(CACHE, f"{name}_{key}.npz")
    if os.path.exists(path):
        with np.load(path) as stored:
            return {k: stored[k] for k in stored.files}
    result = function(**params)
    np.savez_compressed(path, **result)
    return {k: np.asarray(v) for k, v in result.items()}


def write_csv(name, columns, header_note=None):
    """Write equal-length columns to ``outputs/data/<name>.csv``.

    Parameters
    ----------
    columns : dict
        Column name (with units) mapped to a 1-D array.
    header_note : str, optional
        One line of context written as a ``#`` comment above the header.
    """
    ensure_dirs()
    names = list(columns)
    table = np.column_stack([np.asarray(columns[k], dtype=float)
                             for k in names])
    path = os.path.join(DATA, f"{name}.csv")
    with open(path, "w") as handle:
        if header_note:
            handle.write(f"# {header_note}\n")
        handle.write(",".join(names) + "\n")
        np.savetxt(handle, table, delimiter=",", fmt="%.10g")
    return path


def write_report(name, lines):
    """Write a plain-text report to ``outputs/reports/<name>.txt``."""
    ensure_dirs()
    path = os.path.join(REPORTS, f"{name}.txt")
    with open(path, "w") as handle:
        handle.write("\n".join(lines).rstrip() + "\n")
    return path
