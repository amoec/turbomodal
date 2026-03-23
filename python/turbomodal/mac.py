"""Modal Assurance Criterion (MAC) utilities.

Provides vectorized MAC computation for comparing mode shapes,
used by mode tracking and result validation.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def compute_mac(
    phi_a: NDArray,
    phi_b: NDArray,
) -> NDArray:
    """Compute MAC matrix between two sets of mode shapes.

    Parameters
    ----------
    phi_a : (n_modes_a, n_dof) or (n_dof,) complex mode shapes.
    phi_b : (n_modes_b, n_dof) or (n_dof,) complex mode shapes.

    Returns
    -------
    MAC : (n_modes_a, n_modes_b) MAC values in [0, 1].
        ``MAC[i, j] = |phi_a[i]^H phi_b[j]|^2 / (||phi_a[i]||^2 * ||phi_b[j]||^2)``
    """
    a = np.atleast_2d(np.asarray(phi_a, dtype=np.complex128))
    b = np.atleast_2d(np.asarray(phi_b, dtype=np.complex128))

    # Norms squared: (n_modes,)
    norms_a = np.real(np.sum(a * np.conj(a), axis=1))
    norms_b = np.real(np.sum(b * np.conj(b), axis=1))

    # Cross terms: (n_modes_a, n_modes_b)
    cross = np.abs(a @ b.conj().T) ** 2

    denom = np.outer(norms_a, norms_b)
    return cross / np.maximum(denom, 1e-300)


def compute_auto_mac(phi: NDArray) -> NDArray:
    """Compute auto-MAC (mode shapes compared against themselves).

    For well-separated modes, off-diagonal values should be near zero.

    Parameters
    ----------
    phi : (n_modes, n_dof) or (n_dof,) complex mode shapes.

    Returns
    -------
    MAC : (n_modes, n_modes) auto-MAC matrix.
    """
    return compute_mac(phi, phi)
