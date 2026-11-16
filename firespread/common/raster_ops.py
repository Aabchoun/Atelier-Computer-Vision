"""Small neighbourhood operations on the grid (used by the cellular automaton and the fusion)."""

from __future__ import annotations

import numpy as np

# (d_row, d_col) offsets of the 8 neighbours (Moore neighbourhood)
NEIGHBOURS_8 = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))


def shift(array: np.ndarray, d_row: int, d_col: int, fill=0) -> np.ndarray:
    """out[r, c] = array[r - d_row, c - d_col]; cells coming from outside the array get `fill`."""
    out = np.full_like(array, fill)
    h, w = array.shape
    src_r = slice(max(0, -d_row), h - max(0, d_row))
    dst_r = slice(max(0, d_row), h - max(0, -d_row))
    src_c = slice(max(0, -d_col), w - max(0, d_col))
    dst_c = slice(max(0, d_col), w - max(0, -d_col))
    out[dst_r, dst_c] = array[src_r, src_c]
    return out


def dilate(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    """Binary dilation with the 8-neighbourhood."""
    out = np.asarray(mask, dtype=bool).copy()
    for _ in range(iterations):
        current = out.copy()
        for d_row, d_col in NEIGHBOURS_8:
            out |= shift(current, d_row, d_col, fill=False)
    return out
