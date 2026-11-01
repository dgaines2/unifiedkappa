"""End-to-end checks against the reference outputs in examples/."""

import numpy as np
import pytest

from unifiedkappa import ShengBTEAnalyzer, UnifiedkappaManager

# reference .dat files are written with 8 decimals
ATOL = 1e-6


@pytest.mark.parametrize("temperature", [300, 600, 900])
def test_minikappa_phonopy_example(phonopy_example_dir, tmp_path, temperature):
    manager = UnifiedkappaManager.from_parameters(
        poscar_path=phonopy_example_dir / "POSCAR-prim",
        supercell_matrix=np.eye(3) * 4,
        primitive_matrix=np.eye(3),
        force_constants_filename=phonopy_example_dir / "FORCE_CONSTANTS",
        mesh=[12, 12, 12],
    )
    results = manager.run_minikappa(
        temperatures=[float(temperature)], output_dir=tmp_path, verbose=False
    )
    ref = np.loadtxt(phonopy_example_dir / f"minikappa-{temperature}-2.0.dat")
    out = np.loadtxt(tmp_path / f"minikappa-{temperature}-2.0.dat")
    np.testing.assert_allclose(out, ref, atol=ATOL)
    kappa = results[float(temperature)][2.0]
    np.testing.assert_allclose(kappa["F"].flatten(), ref[2], atol=ATOL)


@pytest.fixture(scope="module")
def shengbte_run(tmp_path_factory, shengbte_example_dir):
    """Run unifiedkappa + minikappa once on a copy of the ShengBTE example."""
    import shutil

    workdir = tmp_path_factory.mktemp("sbte") / "shengbte_example"
    shutil.copytree(shengbte_example_dir, workdir)
    (workdir / "T300K" / "unifiedkappa.dat").unlink()
    (workdir / "minikappa-300-2.0.dat").unlink()
    manager = UnifiedkappaManager.from_control(workdir, poscar_filename="POSCAR-prim")
    unified = manager.run_unified_kappa(verbose=False)
    mini = manager.run_minikappa(
        temperatures=[300.0], tau_factors=[2.0], output_dir=workdir, verbose=False
    )
    return workdir, unified, mini


@pytest.mark.slow
def test_unified_kappa_shengbte_example(shengbte_run, shengbte_example_dir):
    workdir, _, _ = shengbte_run
    ref = np.loadtxt(shengbte_example_dir / "T300K" / "unifiedkappa.dat")
    out = np.loadtxt(workdir / "T300K" / "unifiedkappa.dat")
    np.testing.assert_allclose(out, ref, atol=ATOL)


@pytest.mark.slow
def test_minikappa_shengbte_example(shengbte_run, shengbte_example_dir):
    workdir, _, _ = shengbte_run
    ref = np.loadtxt(shengbte_example_dir / "minikappa-300-2.0.dat")
    out = np.loadtxt(workdir / "minikappa-300-2.0.dat")
    np.testing.assert_allclose(out, ref, atol=ATOL)


@pytest.mark.slow
def test_diagonal_reproduces_shengbte_rta(shengbte_run, sbte):
    """With ShengBTE's own v and RTA rates, kappaD must equal ShengBTE's RTA kappa."""
    _, unified, _ = shengbte_run
    kappaD = unified[300]["D"]
    np.testing.assert_allclose(
        kappaD.flatten(), sbte.kappa_rta_tensor["300"], rtol=1e-6, atol=1e-6
    )


@pytest.mark.slow
def test_phonopy_mesh_matches_shengbte_fbz(shengbte_example_dir, sbte):
    """run_unified_kappa pairs phonopy's mesh with ShengBTE's FBZ arrays by index."""
    manager = UnifiedkappaManager.from_control(
        shengbte_example_dir, poscar_filename="POSCAR-prim"
    )
    manager.phonon.run_mesh(
        manager.mesh, is_gamma_center=True, is_time_reversal=False, is_mesh_symmetry=False
    )
    mesh = manager.phonon.mesh
    dq = mesh.qpoints - sbte.qpoints_fbz
    np.testing.assert_allclose(dq - np.rint(dq), 0.0, atol=1e-8)
    freqs_sbte = sbte.map_ibz_quantity_to_fbz(sbte.frequencies)  # rad/ps
    np.testing.assert_allclose(mesh.frequencies * 2 * np.pi, freqs_sbte, atol=1e-2)


def test_shengbte_analyzer_reads_new_output(shengbte_workdir):
    """ShengBTEAnalyzer tolerates a run without unifiedkappa.dat."""
    (shengbte_workdir / "T300K" / "unifiedkappa.dat").unlink()
    sbte = ShengBTEAnalyzer(shengbte_workdir)
    assert not hasattr(sbte, "_unified_kappa")
