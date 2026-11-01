"""3D bar plots of the frequency-resolved unified kappa and minikappa.

Reads the histograms written by run.py and saves, next to each histogram file,
{prefix}-{d,od}_xx.pdf: the diagonal and off-diagonal contributions to kappa_xx
binned by the frequencies of the two phonon modes. Requires matplotlib
(pip install "unifiedkappa[plot]").
"""

from pathlib import Path

from unifiedkappa.plotting import make_kappa_histogram_plots

shengbte_dir = Path(__file__).resolve().parent

for filename_prefix in [
    shengbte_dir / "T300K" / "unifiedkappa",
    shengbte_dir / "minikappa-300-2.0",
]:
    make_kappa_histogram_plots(filename_prefix, parts=("d", "od"), directions=("xx",))
