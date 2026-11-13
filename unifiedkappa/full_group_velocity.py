# Copyright (c) Yi Xia and Dale Gaines II
# Distributed under the terms of the BSD 3-Clause License

"""Off-diagonal (full) phonon group velocities.

    v_ij(q) = factor^2 <e_i| dD/dq |e_j> / (f_i + f_j),   f_i, f_j > cutoff

Degenerate subspaces are rotated so that dD along the [1, 2, 3] Cartesian
direction is diagonal in each.
"""

import numpy as np
from phonopy.harmonic.derivative_dynmat import DerivativeOfDynamicalMatrix
from phonopy.harmonic.dynamical_matrix import DynamicalMatrixGL
from phonopy.phonon.degeneracy import DEFAULT_CUTOFF

_DIRECTIONS = np.array([[1, 2, 3], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype="double")
_DIRECTIONS[0] /= np.linalg.norm(_DIRECTIONS[0])


def degenerate_sets(freqs, cutoff=DEFAULT_CUTOFF):
    """Indices of degenerate bands, grouped; freqs must be in ascending order.

    For ascending freqs these are the sets of phonopy's degenerate_sets (bands
    within cutoff of a member of the set), which is O(nband^2) in Python: a
    set is a run of bands whose consecutive differences are below cutoff.
    Neither version is correct for other orders, so those raise ValueError.
    """
    gaps = np.diff(freqs)
    if np.any(gaps < 0):
        raise ValueError("freqs must be in ascending order, as eigh returns them")
    breaks = np.flatnonzero(gaps >= cutoff) + 1
    return np.split(np.arange(len(freqs)), breaks)


class FullGroupVelocity:
    """Full group velocity matrices v_ij(q) of shape (nband, nband, 3).

    Args:
        phonon: phonopy.Phonopy with force constants set.
        q_length: None for the analytical derivative of D(q); a float for a
            central finite difference with this step (1/Angstrom). "auto"
            uses 1e-5 with Gonze-Lee NAC and the analytical derivative
            otherwise.
        cutoff_frequency: modes at or below this frequency (THz) get zero
            velocity, as in phonopy's GroupVelocity (default 1e-4).
    """

    def __init__(self, phonon, q_length="auto", cutoff_frequency=1e-4):
        self._dynmat = phonon.dynamical_matrix
        self._factor = phonon.unit_conversion_factor
        self._cutoff_frequency = cutoff_frequency
        self._cell = self._dynmat.primitive.cell
        if q_length == "auto":
            q_length = 1e-5 if isinstance(self._dynmat, DynamicalMatrixGL) else None
        self._q_length = q_length
        self._ddm = (
            DerivativeOfDynamicalMatrix(self._dynmat) if q_length is None else None
        )

    def _dynamical_matrix(self, q):
        self._dynmat.run(q)
        return self._dynmat.dynamical_matrix

    def _derivatives(self, q):
        """dD/dq along _DIRECTIONS (Cartesian), shape (4, n, n)."""
        if self._ddm is not None:
            self._ddm.run(q)
            ddm = self._ddm.d_dynamical_matrix  # (3, n, n), Cartesian
            return np.einsum("dj,jab->dab", _DIRECTIONS, ddm)
        ddms = []
        for dq_cart in _DIRECTIONS * self._q_length:
            dq = np.dot(self._cell, dq_cart)  # Cartesian -> fractional
            ddms.append(
                (self._dynamical_matrix(q + dq) - self._dynamical_matrix(q - dq))
                / self._q_length
                / 2
            )
        return np.array(ddms)

    def run_q(self, q):
        """Return (frequencies [THz], v_full [nband, nband, 3]) at q."""
        q = np.asarray(q, dtype="double")
        eigvals, eigvecs = np.linalg.eigh(self._dynamical_matrix(q))
        eigvals = eigvals.real
        freqs = np.sqrt(np.abs(eigvals)) * np.sign(eigvals) * self._factor

        ddms = self._derivatives(q)
        for deg in degenerate_sets(freqs):
            if len(deg) < 2:
                continue
            eigsets = eigvecs[:, deg]
            _, rotation = np.linalg.eigh(eigsets.T.conj() @ ddms[0] @ eigsets)
            eigvecs[:, deg] = eigsets @ rotation

        v_full = (eigvecs.conj().T @ (ddms[1:4] @ eigvecs)).transpose(1, 2, 0)
        above = freqs > self._cutoff_frequency
        mask = above[:, None] & above[None, :]
        denom = freqs[:, None] + freqs[None, :]
        scale = np.where(mask, self._factor**2 / np.where(mask, denom, 1.0), 0.0)
        return freqs, v_full * scale[..., None]

    def run(self, qpoints):
        """Return (frequencies [nq, nband], v_full [nq, nband, nband, 3])."""
        results = [self.run_q(q) for q in qpoints]
        freqs = np.array([r[0] for r in results])
        v_full = np.array([r[1] for r in results], dtype=np.complex128)
        return freqs, v_full
