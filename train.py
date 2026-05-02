"""Compat shim — keep `python train.py` working after the package move.

The real entry point is :mod:`liminal.train`. This file exists so that
the original invocation pattern in the README continues to work for
people who clone and run from the repo root before installing.
"""

from liminal.train import main

if __name__ == "__main__":
    main()
