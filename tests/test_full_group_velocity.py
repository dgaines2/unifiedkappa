import numpy as np
import pytest
from phonopy.phonon.group_velocity import GroupVelocity

from unifiedkappa import FullGroupVelocity

# general q-points (no band degeneracies) and Gamma
GENERAL_QPOINTS = np.array([[0.1, 0.2, 0.3], [0.37, -0.11, 0.05], [0.45, 0.21, 0.13]])


@pytest.fixture(scope="module")
def fgv(nacl_phonon):
    return FullGroupVelocity(nacl_phonon)


def test_shapes(fgv):
    freqs, v_full = fgv.run(GENERAL_QPOINTS)
    assert freqs.shape == (3, 6)
    assert v_full.shape == (3, 6, 6, 3)
    assert v_full.dtype == np.complex128


def test_frequencies_match_phonopy(fgv, nacl_phonon):
    freqs, _ = fgv.run(GENERAL_QPOINTS)
    nacl_phonon.run_qpoints(GENERAL_QPOINTS)
    np.testing.assert_allclose(freqs, nacl_phonon.qpoints.frequencies, atol=1e-8)


def test_hermitian(fgv):
    """v_ij = conj(v_ji) since dD/dq is Hermitian and the prefactor is symmetric."""
    _, v_full = fgv.run(GENERAL_QPOINTS)
    np.testing.assert_allclose(
        v_full, v_full.conj().transpose(0, 2, 1, 3), atol=1e-10 * np.abs(v_full).max()
    )


def test_diagonal_matches_phonopy_group_velocity(fgv, nacl_phonon):
    """At non-degenerate q-points the diagonal is phonopy's group velocity."""
    _, v_full = fgv.run(GENERAL_QPOINTS)
    gv = GroupVelocity(nacl_phonon.dynamical_matrix, symmetry=None)
    gv.run(GENERAL_QPOINTS)
    diag = np.einsum("qiik->qik", v_full)
    np.testing.assert_allclose(diag.imag, 0.0, atol=1e-10)
    np.testing.assert_allclose(diag.real, gv.group_velocities, rtol=1e-6, atol=1e-8)


def test_finite_difference_matches_analytical(nacl_phonon, fgv):
    fd = FullGroupVelocity(nacl_phonon, q_length=1e-5)
    _, v_analytical = fgv.run(GENERAL_QPOINTS)
    _, v_fd = fd.run(GENERAL_QPOINTS)
    np.testing.assert_allclose(v_fd, v_analytical, atol=1e-5 * np.abs(v_analytical).max())


def test_cutoff_frequency_zeros_acoustic_modes(nacl_phonon):
    # NaCl acoustic modes at Gamma are ~5e-4 THz (FCs slightly violate the ASR)
    cutoff = 1e-2
    freqs, v_full = FullGroupVelocity(nacl_phonon, cutoff_frequency=cutoff).run_q(
        [0.0, 0.0, 0.0]
    )
    acoustic = freqs <= cutoff
    assert acoustic.sum() == 3
    assert np.all(v_full[acoustic] == 0)
    assert np.all(v_full[:, acoustic] == 0)
