"""Minikappa for NaCl from harmonic force constants only (no ShengBTE run).

Writes minikappa-{300,600,900}-2.0.dat
"""

from pathlib import Path

import numpy as np

from unifiedkappa import UnifiedkappaManager

phonopy_dir = Path(__file__).resolve().parent

manager = UnifiedkappaManager.from_parameters(
    poscar_path=phonopy_dir / "POSCAR-prim",
    supercell_matrix=np.eye(3) * 4,
    primitive_matrix=np.eye(3),
    force_constants_filename=phonopy_dir / "FORCE_CONSTANTS",
    mesh=[12, 12, 12],
)
minikappa_results = manager.run_minikappa(
    temperatures=[300.0, 600.0, 900.0],
    tau_factors=[2.0],
    output_dir=phonopy_dir,
)
