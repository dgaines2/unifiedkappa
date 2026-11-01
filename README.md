# Unifiedkappa
Scripts for computing the diagonal and off-diagonal contributions to lattice thermal conductivity, based on the inputs and outputs from ShengBTE. Off-diagonal group velocity tensors, with degeneracy properly treated, are computed by `unifiedkappa/full_group_velocity.py`.
# How to use the scripts
* Go to the folder `unifiedkappa/shengbte_example` and run `python ../unified_kappa.py`
  - This folder contains the inputs and outputs typically found in ShengBTE calculations
  - Minikappa outputs are saved in the base ShengBTE (or phonopy) directory as they're based on only harmonic properties
  - Unifiedkappa outputs are saved in each ShengBTE temperature directory (e.g. T300K) as they're based on anharmonic phonon scattering rates
* Note that in order to read scattering rates produced by ShengBTE, you must explicitly specify `onlyharmonic=.FALSE.` in the `CONTROL` file
