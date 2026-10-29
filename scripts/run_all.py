"""Reproduce every figure, animation and report from scratch."""

import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
STEPS = ["precompute", "fig00_lens", "fig01_modes", "fig02_tide",
         "fig03_transfer", "fig04_rectification", "fig05_setup",
         "fig06_seasonal", "fig07_regime", "fig08_verification",
         "anim01_tide", "anim02_season", "make_reports"]


def main():
    """Run the scripts in order; cached simulations are not repeated."""
    start = time.time()
    for name in STEPS:
        tic = time.time()
        subprocess.run([sys.executable, os.path.join(HERE, f"{name}.py")],
                       check=True)
        print(f"{name:22s} {time.time() - tic:7.1f} s", flush=True)
    print(f"all done in {(time.time() - start) / 60:.1f} min")


if __name__ == "__main__":
    main()
