"""Compat shim — keep `python evaluate.py` working after the package move.

The real entry point is :mod:`liminal.evaluate`.
"""

from liminal.evaluate import main

if __name__ == "__main__":
    main()
