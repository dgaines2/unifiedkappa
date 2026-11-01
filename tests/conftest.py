"""Shared fixtures for the unifiedkappa test suite.

Tests use the data in examples/ (NaCl, 4x4x4 supercell):
  - examples/phonopy_example: POSCAR-prim + FORCE_CONSTANTS, reference minikappa
    outputs for a 12x12x12 mesh
  - examples/shengbte_example: a complete ShengBTE/FourPhonon run (25x25x25 mesh,
    T=300 K), reference minikappa and unifiedkappa outputs
"""

import shutil
from pathlib import Path

import numpy as np
import phonopy
import pytest

from unifiedkappa import ShengBTEAnalyzer

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"
PHONOPY_EXAMPLE_DIR = EXAMPLES_DIR / "phonopy_example"
SHENGBTE_EXAMPLE_DIR = EXAMPLES_DIR / "shengbte_example"


@pytest.fixture(scope="session")
def phonopy_example_dir():
    return PHONOPY_EXAMPLE_DIR


@pytest.fixture(scope="session")
def shengbte_example_dir():
    return SHENGBTE_EXAMPLE_DIR


@pytest.fixture
def shengbte_workdir(tmp_path):
    """Writable copy of the ShengBTE example (run_unified_kappa writes into T*K/)."""
    workdir = tmp_path / "shengbte_example"
    shutil.copytree(SHENGBTE_EXAMPLE_DIR, workdir)
    return workdir


@pytest.fixture(scope="session")
def sbte():
    """Read-only ShengBTEAnalyzer for the ShengBTE example."""
    return ShengBTEAnalyzer(SHENGBTE_EXAMPLE_DIR)


@pytest.fixture(scope="session")
def nacl_phonon():
    """NaCl phonopy object with force constants from the phonopy example."""
    return phonopy.load(
        supercell_matrix=np.eye(3, dtype=int) * 4,
        primitive_matrix=np.eye(3),
        unitcell_filename=str(PHONOPY_EXAMPLE_DIR / "POSCAR-prim"),
        force_constants_filename=str(PHONOPY_EXAMPLE_DIR / "FORCE_CONSTANTS"),
        is_symmetry=False,
    )
