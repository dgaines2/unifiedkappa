import numpy as np
import pytest

from unifiedkappa import ShengBTEAnalyzer, unwrap_qpoints


def test_control(sbte):
    assert sbte.nelements == 2
    assert sbte.natoms == 2
    assert sbte.nbands == 6
    assert sbte.elements == ["Na", "Cl"]
    np.testing.assert_array_equal(sbte.ngrid, [25, 25, 25])
    np.testing.assert_array_equal(sbte.scell_matrix, np.diag([4, 4, 4]))
    np.testing.assert_array_equal(sbte.temperatures, [300])
    assert sbte.four_phonon
    assert not sbte.onlyharmonic
    # lattvec in nm (lfactor=0.1); fcc with a/2 = 0.28016 nm
    assert sbte.volume == pytest.approx(2 * 0.28016230200**3, rel=1e-10)


def test_qpoints(sbte):
    assert sbte.nq_fbz == 25**3
    assert sbte.qpoints_fbz.shape == (25**3, 3)
    assert sbte.qpoints.shape == (sbte.nq, 3)
    # every FBZ point maps to exactly one IBZ point
    assert sbte.qpoint_degeneracy.sum() == sbte.nq_fbz
    assert sbte.qpoint_weights.sum() == pytest.approx(1.0)
    # degeneracies agree with the third column ShengBTE writes in BTE.qpoints
    bte_qpoints = np.loadtxt(sbte.workdir / "BTE.qpoints")
    np.testing.assert_array_equal(sbte.qpoint_degeneracy, bte_qpoints[:, 2])


def test_array_shapes(sbte):
    assert sbte.frequencies.shape == (sbte.nq, 6)
    assert sbte.group_velocity.shape == (sbte.nq, 6, 3)
    assert sbte.group_velocity_fbz.shape == (sbte.nq_fbz, 6, 3)
    assert sbte.rates_rta["300"].shape == (sbte.nq, 6)
    assert sbte.rates_rta_fbz["300"].shape == (sbte.nq_fbz, 6)


def test_map_ibz_quantity_to_fbz(sbte):
    freqs_fbz = sbte.map_ibz_quantity_to_fbz(sbte.frequencies)
    for i_ibz, idx_fbz in enumerate(sbte.idx_fbz_to_ibz):
        expected = np.broadcast_to(sbte.frequencies[i_ibz], (len(idx_fbz), 6))
        np.testing.assert_array_equal(freqs_fbz[idx_fbz], expected)


def test_scattering_rate_cutoff(sbte, shengbte_example_dir):
    # with the default cutoff of 0, zero/NaN rates (acoustic modes at Gamma) become 1e10
    assert np.all(sbte.rates_rta["300"] > 0)
    assert np.all(np.isfinite(sbte.rates_rta["300"]))
    raw = ShengBTEAnalyzer(shengbte_example_dir, scattering_rate_cutoff=None)
    gamma_rates = raw.rates_rta["300"][0, :3]
    assert np.all((gamma_rates == 0) | np.isnan(gamma_rates))
    np.testing.assert_array_equal(sbte.rates_rta["300"][0, :3], 1e10)


def test_kappa(sbte):
    kappa_tensor = np.loadtxt(sbte.workdir / "T300K" / "BTE.kappa_tensor")
    rta, converged = kappa_tensor[0, 1:], kappa_tensor[-1, 1:]
    np.testing.assert_allclose(sbte.kappa_rta_tensor["300"], rta, atol=1e-5)
    np.testing.assert_allclose(sbte.kappa_tensor["300"], converged, atol=1e-5)
    assert sbte.kappa_rta["300"] == pytest.approx(6.246154555)
    assert sbte.kappa["300"] > sbte.kappa_rta["300"]


def test_unified_kappa(sbte):
    ref = np.loadtxt(sbte.workdir / "T300K" / "unifiedkappa.dat")
    np.testing.assert_allclose(sbte.unified_kappa_D_tensor["300"], ref[0], atol=1e-5)
    np.testing.assert_allclose(sbte.unified_kappa_OD_tensor["300"], ref[1], atol=1e-5)
    np.testing.assert_allclose(sbte.unified_kappa_tensor["300"], ref[2], atol=1e-5)
    assert sbte.unified_kappa["300"] == pytest.approx(
        sbte.unified_kappa_D["300"] + sbte.unified_kappa_OD["300"], abs=1e-5
    )


def test_missing_bte_out(tmp_path):
    with pytest.raises(FileNotFoundError, match="No BTE.out"):
        ShengBTEAnalyzer(tmp_path)


def test_incomplete_run(tmp_path):
    (tmp_path / "BTE.out").write_text("still running\n")
    with pytest.raises(RuntimeError, match="BTE calculation failed"):
        ShengBTEAnalyzer(tmp_path)


def test_unwrap_qpoints():
    qpoints = np.array([[0.0, -0.25, 0.5], [-0.04, 0.12, -0.48]])
    expected = np.array([[0.0, 0.75, 0.5], [0.96, 0.12, 0.52]])
    np.testing.assert_allclose(unwrap_qpoints(qpoints), expected)
