"""Shared utilities for turbomodal Python modules."""

from __future__ import annotations

import numpy as np


def progress_bar(
    current: int,
    total: int,
    width: int = 40,
    prefix: str = "",
    suffix: str = "",
    elapsed: float = 0.0,
) -> str:
    """Render a compact text progress bar.

    Parameters
    ----------
    current : current step (1-based)
    total : total number of steps
    width : bar width in characters
    prefix : text before the bar
    suffix : text after the percentage
    elapsed : elapsed time in seconds (for ETA calculation)
    """
    frac = current / max(total, 1)
    filled = int(width * frac)
    bar = "=" * filled + ">" * (1 if filled < width else 0) + "." * (width - filled - 1)
    pct = f"{100 * frac:5.1f}%"

    eta_str = ""
    if current > 0 and elapsed > 0:
        eta = elapsed / current * (total - current)
        if eta >= 60:
            eta_str = f"  ETA {eta / 60:.1f}m"
        else:
            eta_str = f"  ETA {eta:.0f}s"

    return f"\r{prefix}[{bar}] {pct}  ({current}/{total}){suffix}{eta_str}"


def rotation_matrix_3x3(theta: float, axis: int) -> np.ndarray:
    """Build a 3x3 rotation matrix for angle *theta* about the given axis.

    Parameters
    ----------
    theta : rotation angle in radians
    axis : 0=X, 1=Y, 2=Z
    """
    c = np.cos(theta)
    s = np.sin(theta)
    if axis == 0:  # X-axis: rotate in YZ
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == 1:  # Y-axis: rotate in XZ
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    # Z-axis: rotate in XY
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def rotation_matrix_4x4_flat(axis: int, angle: float) -> list[float]:
    """Build a flat row-major 4x4 affine rotation matrix (for gmsh setPeriodic).

    Parameters
    ----------
    axis : 0=X, 1=Y, 2=Z
    angle : rotation angle in radians
    """
    R = rotation_matrix_3x3(angle, axis)
    # Build 4x4 homogeneous: [[R, 0], [0, 1]]
    out = []
    for i in range(3):
        for j in range(3):
            out.append(float(R[i, j]))
        out.append(0.0)
    out.extend([0.0, 0.0, 0.0, 1.0])
    return out
