import numpy as np
import pytest

from unifiedkappa import UnifiedkappaManager, read_minikappa_file

# constants used by calculate_unified_kappa (hbar in J*ps, kB in J/K)
HBAR = 1.054571726470000e-022
KB = 1.380648813000000e-023


@pytest.fixture
def manager(nacl_phonon):
    return UnifiedkappaManager(nacl_phonon, mesh=[4, 4, 4], symprec=None)


@pytest.fixture
def synthetic_inputs():
    rng = np.random.default_rng(0)
    nqpt, nband = 20, 6
    freqs = rng.uniform(1.0, 40.0, size=(nqpt, nband))  # rad/ps
    Gamma = rng.uniform(0.1, 5.0, size=(nqpt, nband))  # 1/ps
    v = rng.normal(size=(nqpt, nband, nband, 3)) + 1j * rng.normal(
        size=(nqpt, nband, nband, 3)
    )
    gvfull = (v + v.conj().transpose(0, 2, 1, 3)) / 2  # Hermitian, km/s
    return freqs, Gamma, gvfull


def test_maximum_scattering_rates():
    freqs = np.array([[-1.0, 0.0, 2 * np.pi, 4 * np.pi]])
    Gamma = UnifiedkappaManager.get_maximum_scattering_rates(freqs, tau_factor=2.0)
    np.testing.assert_allclose(Gamma, [[1e10, 1e10, 2.0, 4.0]])


def test_total_is_sum_of_parts(manager, synthetic_inputs):
    kappaD, kappaOD, kappaF = manager.calculate_unified_kappa(*synthetic_inputs)
    np.testing.assert_allclose(kappaF, kappaD + kappaOD, rtol=1e-12)
    assert np.all(np.diag(kappaD) > 0)


def test_diagonal_is_rta(manager, synthetic_inputs):
    """With no off-diagonal velocities, kappa = sum_q,s C_s v_s v_s / Gamma_s / (V N)."""
    freqs, Gamma, gvfull = synthetic_inputs
    nqpt, nband = freqs.shape
    v_diag = np.einsum("qiik->qik", gvfull).real
    gvfull_diag = np.zeros_like(gvfull)
    idx = np.arange(nband)
    gvfull_diag[:, idx, idx] = v_diag

    T = 300.0
    kappaD, kappaOD, _ = manager.calculate_unified_kappa(
        freqs, Gamma, gvfull_diag, temperature=T
    )
    np.testing.assert_array_equal(kappaOD, 0.0)

    x = HBAR * freqs / (KB * T)
    n = 1.0 / np.expm1(x)
    heat_capacity = KB * x**2 * n * (n + 1)  # J/K per mode
    volume = manager.phonon.primitive.volume / 1000.0  # nm^3
    expected = np.einsum("qs,qsa,qsb->ab", heat_capacity / Gamma, v_diag, v_diag)
    expected *= 1e21 / (volume * nqpt)  # -> W/m/K
    np.testing.assert_allclose(kappaD, expected, rtol=1e-10)


def test_frequency_cutoff(manager, synthetic_inputs):
    freqs, Gamma, gvfull = synthetic_inputs
    kappaD, kappaOD, kappaF = manager.calculate_unified_kappa(
        freqs, Gamma, gvfull, freqcf=freqs.max() + 1
    )
    np.testing.assert_array_equal(kappaF, 0.0)


@pytest.mark.parametrize("batch_size", [1, 7, None])
def test_batching_is_invariant(nacl_phonon, synthetic_inputs, batch_size):
    ref = UnifiedkappaManager(nacl_phonon, symprec=None, batch_size=10**6)
    batched = UnifiedkappaManager(nacl_phonon, symprec=None, batch_size=batch_size)
    for a, b in zip(
        ref.calculate_unified_kappa(*synthetic_inputs),
        batched.calculate_unified_kappa(*synthetic_inputs),
    ):
        np.testing.assert_allclose(a, b, rtol=1e-12)


def test_symmetrization_cubic(nacl_phonon, synthetic_inputs):
    """NaCl is cubic, so symmetrized tensors must be isotropic."""
    manager = UnifiedkappaManager(nacl_phonon, symprec=1e-3)
    for kappa in manager.calculate_unified_kappa(*synthetic_inputs):
        np.testing.assert_allclose(kappa, np.eye(3) * kappa[0, 0], atol=1e-12)


def test_save_histogram(nacl_phonon, synthetic_inputs, tmp_path):
    manager = UnifiedkappaManager(
        nacl_phonon, symprec=None, save_histogram=True, n_histogram_bins=10
    )
    prefix = tmp_path / "hist"
    kappaD, kappaOD, _ = manager.calculate_unified_kappa(
        *synthetic_inputs, filename_prefix=str(prefix)
    )
    for part, kappa in [("d", kappaD), ("od", kappaOD)]:
        for i, direction in enumerate(["xx", "yy", "zz"]):
            hist = np.loadtxt(f"{prefix}-{part}_{direction}.txt")
            assert hist.shape == (10, 10)
            assert hist.sum() == pytest.approx(kappa[i, i], rel=1e-8)


def test_run_unified_kappa_requires_shengbte_dir(manager):
    with pytest.raises(RuntimeError, match="shengbte_dir"):
        manager.run_unified_kappa(verbose=False)


def test_read_minikappa_file(phonopy_example_dir):
    kappaD, kappaOD, kappaF = read_minikappa_file(
        phonopy_example_dir / "minikappa-300-2.0.dat"
    )
    assert kappaD[0, 0] == pytest.approx(0.31185308)
    assert kappaOD[0, 0] == pytest.approx(0.19198264)
    np.testing.assert_allclose(kappaF, kappaD + kappaOD, atol=1e-7)


def test_from_control(shengbte_example_dir):
    manager = UnifiedkappaManager.from_control(
        shengbte_example_dir, poscar_filename="POSCAR-prim"
    )
    np.testing.assert_array_equal(manager.mesh, [25, 25, 25])
    np.testing.assert_array_equal(manager.phonon.supercell_matrix, np.diag([4, 4, 4]))
    assert manager.shengbte_dir == shengbte_example_dir
