"""Run every simulation once and cache the results in outputs/cache."""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(
    __file__))))

from coastal_interface import experiments  # noqa: E402


def main():
    """Fill the cache; later scripts only read from it."""
    start = time.time()
    experiments.precompute()
    print(f"precompute finished in {(time.time() - start) / 60:.1f} min")


if __name__ == "__main__":
    main()
