"""Plots of frequency-resolved thermal conductivity.

UnifiedkappaManager(save_histogram=True) bins the diagonal ("d") and off-diagonal
("od") contributions to kappa by the frequencies (omega_i, omega_j) of the two modes and
writes one file per part and direction: {filename_prefix}-{part}_{direction}.txt, e.g.
T300K/unifiedkappa-od_xx.txt. Summing a histogram gives that component of kappa
before symmetrization: the tensors in the .dat files are symmetrized (symprec), the
histograms are not, so e.g. for a cubic crystal the .dat value of kappa_xx is the mean of
the xx, yy, and zz histogram sums.

Requires matplotlib (pip install "unifiedkappa[plot]").
"""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PARTS = {"d": "Diagonal", "od": "Off-diagonal"}
DIRECTIONS = ("xx", "yy", "zz")
COLORS = {"d": "#009ADE", "od": "#00CD6C"}  # blue, green

_DELTA_FREQ_RE = re.compile(r"delta_freq\s*=\s*([-+0-9.eE]+)")


def get_histogram_path(filename_prefix, part, direction):
    """Path of the histogram file written by UnifiedkappaManager."""
    _check_part_direction(part, direction)
    return Path(f"{filename_prefix}-{part}_{direction}.txt")


def read_kappa_histogram(filename_prefix, part="od", direction="xx", delta_freq=None):
    """
    Read a frequency-resolved kappa histogram.

    Args:
        filename_prefix (str | Path): prefix used when the histogram was written,
            e.g. "T300K/unifiedkappa" or "minikappa-300-2.0"
        part (str): "d" (diagonal) or "od" (off-diagonal)
        direction (str): "xx", "yy", or "zz"
        delta_freq (float | None): bin width in rad/ps. If None, read it from the
            file header (written since the header was added); files without a
            header need it passed explicitly.

    Returns:
        histogram (np.array[nbins, nbins]): contributions in W/m/K; row i and column j
            are the frequency bins of modes i and j, bin n spanning
            [n * delta_freq, (n + 1) * delta_freq)
        delta_freq (float): bin width in rad/ps
    """
    path = get_histogram_path(filename_prefix, part, direction)
    histogram = np.loadtxt(path, ndmin=2)
    if delta_freq is None:
        with open(path) as fr:
            first_line = fr.readline()
        match = _DELTA_FREQ_RE.search(first_line) if first_line.startswith("#") else None
        if match is None:
            raise ValueError(
                f"{path} has no delta_freq header; pass delta_freq (rad/ps) explicitly"
            )
        delta_freq = float(match.group(1))
    return histogram, delta_freq


def make_kappa_histogram_plot(
    histogram,
    delta_freq,
    part="od",
    direction="xx",
    fig_path="kappa_histogram.pdf",
    color=None,
    elev=30,
    azim=60,
    show=False,
    return_fig=False,
    verbose=True,
):
    """
    3D bar plot of a frequency-resolved kappa histogram.

    Args:
        histogram (np.array[nbins, nbins]): contributions in W/m/K
            (see read_kappa_histogram)
        delta_freq (float): bin width in rad/ps
        part (str): "d" or "od", used for the title and default color
        direction (str): "xx", "yy", or "zz", used for the axis label
        fig_path (str | Path | None): path to save the figure. If None, don't save.
        color (str | None): bar color. If None, use COLORS[part].
        elev (float): elevation of the 3D view in degrees
        azim (float): azimuth of the 3D view in degrees
        show (bool): if True, call plt.show()
        return_fig (bool): if True, return the Figure instead of closing it
        verbose (bool): if True, print the save path

    Returns:
        Figure if return_fig is True, otherwise None
    """
    _check_part_direction(part, direction)
    histogram = np.asarray(histogram, dtype=float)
    if histogram.ndim != 2 or histogram.shape[0] != histogram.shape[1]:
        raise ValueError(f"histogram must be square, got shape {histogram.shape}")
    nbins = histogram.shape[0]
    width = delta_freq / (2 * np.pi)  # rad/ps -> THz

    # x: mode j (columns), y: mode i (rows); empty bins are not drawn
    xpos, ypos = np.meshgrid(np.arange(nbins) * width, np.arange(nbins) * width)
    nonzero = histogram != 0
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.view_init(elev=elev, azim=azim)
    if nonzero.any():
        ax.bar3d(
            xpos[nonzero],
            ypos[nonzero],
            np.zeros(nonzero.sum()),
            0.9 * width,
            0.9 * width,
            histogram[nonzero],
            color=COLORS[part] if color is None else color,
            edgecolor="none",
        )
    ax.set_xlim(0, nbins * width)
    ax.set_ylim(0, nbins * width)
    ax.set_xlabel(r"$\nu_j$ (THz)")
    ax.set_ylabel(r"$\nu_i$ (THz)")
    ax.set_zlabel(rf"$\kappa_{{{direction}}}$ (W/m/K)")
    ax.set_title(f"{PARTS[part]}: {histogram.sum():.3f} W/m/K")

    if fig_path is not None:
        fig.savefig(fig_path, bbox_inches="tight")
        if verbose:
            print(f"Saved figure to {fig_path}")
    if show:
        plt.show()
    if return_fig:
        return fig
    plt.close(fig)
    return None


def make_kappa_histogram_plots(
    filename_prefix,
    parts=("d", "od"),
    directions=("xx",),
    fig_format="pdf",
    verbose=True,
    **kwargs,
):
    """
    Read histograms written by UnifiedkappaManager and save a 3D bar plot of each to
    {filename_prefix}-{part}_{direction}.{fig_format}.

    Args:
        filename_prefix (str | Path): prefix used when the histograms were written
        parts (iterable[str]): any of "d", "od"
        directions (iterable[str]): any of "xx", "yy", "zz"
        fig_format (str): figure file extension
        verbose (bool): if True, print the save paths
        **kwargs: passed to make_kappa_histogram_plot

    Returns:
        fig_paths (list[Path]): paths of the saved figures
    """
    fig_paths = []
    for part in parts:
        for direction in directions:
            histogram, delta_freq = read_kappa_histogram(filename_prefix, part, direction)
            fig_path = get_histogram_path(filename_prefix, part, direction).with_suffix(
                f".{fig_format}"
            )
            make_kappa_histogram_plot(
                histogram,
                delta_freq,
                part=part,
                direction=direction,
                fig_path=fig_path,
                verbose=verbose,
                **kwargs,
            )
            fig_paths.append(fig_path)
    return fig_paths


def _check_part_direction(part, direction):
    if part not in PARTS:
        raise ValueError(f"part must be one of {list(PARTS)}, got {part!r}")
    if direction not in DIRECTIONS:
        raise ValueError(
            f"direction must be one of {list(DIRECTIONS)}, got {direction!r}"
        )
