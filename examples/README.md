# Examples
Both examples are rocksalt NaCl with harmonic force constants from a 4x4x4 supercell.
Run them from any directory once `unifiedkappa` is installed:
```bash
python examples/shengbte_example/run.py
python examples/shengbte_example/plot.py   # needs matplotlib: pip install "unifiedkappa[plot]"
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
- `T300K/unifiedkappa-{d,od}_{xx,yy,zz}.txt` and `minikappa-300-2.0-{d,od}_{xx,yy,zz}.txt`:
  the diagonal and off-diagonal contributions binned by the frequencies of the two modes
  (`save_histogram=True`)

Reading scattering rates requires `onlyharmonic=.FALSE.` in `CONTROL`. Takes ~15 s.

`plot.py` reads the histograms (no recalculation) and saves a 3D bar plot of each xx
histogram next to it, e.g. `T300K/unifiedkappa-od_xx.pdf`, using
`unifiedkappa.plotting.make_kappa_histogram_plots`. The histograms are not symmetrized,
unlike the tensors in the `.dat` files, so for cubic NaCl the `.dat` kappa_xx is the mean
of the xx, yy, and zz histogram sums.

## `phonopy_example/`
Harmonic inputs only (`POSCAR-prim`, `FORCE_CONSTANTS`). `run.py` uses
`UnifiedkappaManager.from_parameters` on a 12x12x12 mesh and writes
`minikappa-{300,600,900}-2.0.dat`. Takes ~1 s.

## Output format
Each `.dat` file has three rows, each a flattened 3x3 tensor in W/m/K: the diagonal
(Peierls) part, the off-diagonal (coherence) part, and the total. Read them with
`unifiedkappa.read_minikappa_file`.

Each histogram `.txt` file is an `n_histogram_bins` x `n_histogram_bins` array in W/m/K
whose header gives the bin width `delta_freq` (rad/ps): row i and column j are the
frequency bins of the two modes, bin n spanning [n, n+1) x `delta_freq`. Read them with
`unifiedkappa.plotting.read_kappa_histogram`.

## Reference outputs
The `.dat` and histogram `.txt` files committed here are the reference outputs for the
regression tests in `tests/test_regression.py`. `run.py` overwrites them in place. Off-diagonal tensor
elements are numerical noise (~1e-17) that rounds to `0.00000000` or `-0.00000000`
depending on the run, so `git diff examples/` can show sign-of-zero changes even when the
results are identical; `pytest` compares them numerically.
