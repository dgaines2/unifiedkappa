"""Unified kappa and minikappa for NaCl from a ShengBTE/FourPhonon run.

UnifiedkappaManager.from_control reads the supercell and q-point mesh from CONTROL,
so the phonopy mesh matches ShengBTE's ngrid.

Writes:
  - T300K/unifiedkappa.dat (from ShengBTE's RTA scattering rates)
  - minikappa-300-2.0.dat (from maximum scattering rates, harmonic properties only)
"""

from pathlib import Path

from unifiedkappa import UnifiedkappaManager

shengbte_dir = Path(__file__).resolve().parent

manager = UnifiedkappaManager.from_control(
    shengbte_dir=shengbte_dir,
    poscar_filename="POSCAR-prim",
    force_constants_filename="FORCE_CONSTANTS_2ND",
)
unified_results = manager.run_unified_kappa()
minikappa_results = manager.run_minikappa(
    temperatures=[300.0],
    tau_factors=[2.0],
    output_dir=shengbte_dir,
)
