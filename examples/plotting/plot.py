"""3D bar plots of the frequency-resolved unified kappa and minikappa.

Reads the histograms that ../shengbte_example/run.py writes (save_histogram=True) and
saves the diagonal ("d") and off-diagonal ("od") contributions to kappa_xx, binned by the
frequencies of the two phonon modes, as PDF and PNG files in this directory. Requires
matplotlib (pip install "unifiedkappa[plot]").
"""

from pathlib import Path

import matplotlib.pyplot as plt

from unifiedkappa.plotting import make_kappa_histogram_plot, read_kappa_histogram

plotting_dir = Path(__file__).resolve().parent
shengbte_dir = plotting_dir.parent / "shengbte_example"

plt.rcParams["savefig.dpi"] = 300  # resolution of the PNGs; PDFs are vector graphics

histogram_prefixes = {
    "unifiedkappa-300K": shengbte_dir / "T300K" / "unifiedkappa",
    "minikappa-300K": shengbte_dir / "minikappa-300-2.0",
}

for name, filename_prefix in histogram_prefixes.items():
    for part in ["d", "od"]:
        histogram, delta_freq = read_kappa_histogram(filename_prefix, part, "xx")
        make_kappa_histogram_plot(
            histogram,
            delta_freq,
            part=part,
            direction="xx",
            fig_path=plotting_dir / f"{name}-{part}_xx",
            fig_formats=("pdf", "png"),
        )
