# Unifiedkappa
Computes lattice thermal conductivity, including both the diagonal (particle-like,
Peierls) and off-diagonal (wave-like, coherence) contributions, with a unified
treatment of phonon transport. It supports two kinds of calculation:

* **Minimum thermal conductivity (minikappa)**: needs only harmonic force constants. Every
  phonon mode gets the maximum scattering rate, Γ = `tau_factor` · ω / 2π (the default
  `tau_factor=2` gives Γ = ω/π). This is the model of
  [Xia *et al.*, PNAS **120**, e2302541120 (2023)](https://doi.org/10.1073/pnas.2302541120);
  it bounds the isotropically averaged high-temperature minimum kappa of crystals to
  roughly 0.1–2.6 W/m/K. This package supersedes the original
  [Minikappa](https://github.com/yimavxia/Minikappa) scripts. It produces the same output
  format but uses the public phonopy API instead of a patched phonopy.
* **Unified kappa**: uses the anharmonic phonon scattering rates from a
  ShengBTE/FourPhonon calculation. Diagonal group velocities are taken from ShengBTE
  (`BTE.v_full`) so that they are consistent with its scattering rates, and off-diagonal
  group velocities are computed with phonopy.

Off-diagonal group velocity tensors, with degeneracy properly treated, are computed by
`unifiedkappa/full_group_velocity.py`.

## Installation
```bash
git clone git@github.com:dgaines2/Unifiedkappa.git
cd Unifiedkappa
pip install -e .            # or: uv pip install -e .
```

For development (pytest, ruff, pre-commit, coverage):
```bash
pip install -e ".[dev]"
pre-commit install
```

## Usage
### Minikappa from harmonic force constants
```python
import numpy as np
from unifiedkappa import UnifiedkappaManager

manager = UnifiedkappaManager.from_parameters(
    poscar_path="POSCAR",
    supercell_matrix=np.eye(3) * 4,
    primitive_matrix=np.eye(3),
    force_constants_filename="FORCE_CONSTANTS",
    mesh=[12, 12, 12],
)
minikappa_results = manager.run_minikappa(temperatures=[300.0, 600.0], tau_factors=[2.0])
```

### Unified kappa from a ShengBTE/FourPhonon run
```python
from unifiedkappa import UnifiedkappaManager

# supercell and q-point mesh are read from CONTROL
manager = UnifiedkappaManager.from_control("path/to/shengbte_dir", poscar_filename="POSCAR")
unified_results = manager.run_unified_kappa()
minikappa_results = manager.run_minikappa(temperatures=[300.0])  # also available here
```

See [`examples/`](examples/README.md) for runnable NaCl examples of both.
* Minikappa outputs (`minikappa-{T}-{tau_factor}.dat`) are saved in the base ShengBTE (or
  phonopy) directory as they're based on only harmonic properties
* Unifiedkappa outputs (`T{T}K/unifiedkappa.dat`) are saved in each ShengBTE temperature
  directory as they're based on anharmonic phonon scattering rates
* Each output file holds three flattened 3x3 tensors in W/m/K: diagonal, off-diagonal, and
  total
* Note that in order to read scattering rates produced by ShengBTE, you must explicitly
  specify `onlyharmonic=.FALSE.` in the `CONTROL` file

## Testing
```bash
pytest                 # full suite (~20 s)
pytest -m "not slow"   # skip the full 25x25x25 mesh regression tests
```
The regression tests compare against the reference outputs in `examples/`.

## Citation
If you use the minikappa model, please cite:

> Y. Xia, D. Gaines II, J. He, K. Pal, Z. Li, M. G. Kanatzidis, V. Ozoliņš, and
> C. Wolverton, "A unified understanding of minimum lattice thermal conductivity,"
> *Proc. Natl. Acad. Sci. U.S.A.* **120**, e2302541120 (2023).
> https://doi.org/10.1073/pnas.2302541120

```bibtex
@article{xia_unified_2023,
  title   = {A unified understanding of minimum lattice thermal conductivity},
  author  = {Xia, Yi and Gaines, II, Dale and He, Jiangang and Pal, Koushik and Li, Zhi and
             Kanatzidis, Mercouri G. and Ozoli{\c{n}}{\v{s}}, Vidvuds and Wolverton, Chris},
  journal = {Proceedings of the National Academy of Sciences},
  volume  = {120},
  number  = {26},
  pages   = {e2302541120},
  year    = {2023},
  doi     = {10.1073/pnas.2302541120},
}
```
