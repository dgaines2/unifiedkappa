# Examples
Both examples are rocksalt NaCl with harmonic force constants from a 4x4x4 supercell.
Run them from any directory once `unifiedkappa` is installed:
```bash
python examples/shengbte_example/run.py
python examples/phonopy_example/run.py
```

## `shengbte_example/`
A complete ShengBTE/FourPhonon run (3- and 4-phonon scattering, 25x25x25 mesh,
T = 300 K). `run.py` uses `UnifiedkappaManager.from_control`, which takes the supercell
and q-point mesh from `CONTROL`, and writes:
- `T300K/unifiedkappa.dat`: unified kappa from ShengBTE's RTA scattering rates. Diagonal
  group velocities come from `BTE.v_full`, so they are consistent with ShengBTE's
  scattering rates; off-diagonal velocities come from phonopy
- `minikappa-300-2.0.dat`: minimum kappa from maximum scattering rates (harmonic
  properties only)

Reading scattering rates requires `onlyharmonic=.FALSE.` in `CONTROL`. Takes ~15 s.

## `phonopy_example/`
Harmonic inputs only (`POSCAR-prim`, `FORCE_CONSTANTS`). `run.py` uses
`UnifiedkappaManager.from_parameters` on a 12x12x12 mesh and writes
`minikappa-{300,600,900}-2.0.dat`. Takes ~1 s.

## Output format
Each `.dat` file has three rows, each a flattened 3x3 tensor in W/m/K: the diagonal
(Peierls) part, the off-diagonal (coherence) part, and the total. Read them with
`unifiedkappa.read_minikappa_file`.

## Reference outputs
The `.dat` files committed here are the reference outputs for the regression tests in
`tests/test_regression.py`. `run.py` overwrites them in place. Off-diagonal tensor
elements are numerical noise (~1e-17) that rounds to `0.00000000` or `-0.00000000`
depending on the run, so `git diff examples/` can show sign-of-zero changes even when the
results are identical; `pytest` compares them numerically.
