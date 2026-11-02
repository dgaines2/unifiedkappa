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
from matplotlib.backend_bases import FigureCanvasBase
from matplotlib.colors import LightSource, LinearSegmentedColormap, PowerNorm
from matplotlib.transforms import Bbox
from scipy.ndimage import zoom as upsample_array

PARTS = {"d": "Diagonal", "od": "Off-diagonal"}
DIRECTIONS = ("xx", "yy", "zz")
COLORS = {"d": "#009ADE", "od": "#00CD6C"}  # bar colors: blue, green
# surface colored by height: blue (low) -> white -> yellow -> orange -> red (high),
# after the style of Xia et al., Phys. Rev. X 10, 041029 (2020), Fig. 6(d,e)
SURFACE_CMAP = LinearSegmentedColormap.from_list(
    "unifiedkappa_surface",
    [
        "#1a2fb8",
        "#3f63e8",
        "#a3b6f3",
        "#f3efe4",
        "#f7d04a",
        "#f39a2b",
        "#e0452a",
        "#b2182b",
    ],
)
PLOT_STYLES = ("surface", "bar")

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
    fig_formats=None,
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
        part (str): "d" or "od", used for the default color
        direction (str): "xx", "yy", or "zz", used for the axis label
        fig_path (str | Path | None): path to save the figure. If None, don't save.
        fig_formats (str | iterable[str] | None): if given, save one file per format
            (e.g. ("pdf", "png")) by replacing the extension of fig_path, or appending
            one if fig_path has no figure extension
        color (str | None): bar color. If None, use COLORS[part].
        elev (float): elevation of the 3D view in degrees
        azim (float): azimuth of the 3D view in degrees
        show (bool): if True, call plt.show()
        return_fig (bool): if True, return the Figure instead of closing it
        verbose (bool): if True, print the save paths

    Returns:
        Figure if return_fig is True, otherwise None
    """
    _check_part_direction(part, direction)
    histogram = _check_histogram(histogram)
    nbins = histogram.shape[0]
    width = delta_freq / (2 * np.pi)  # rad/ps -> THz

    # x: mode j (columns), y: mode i (rows); empty bins are not drawn
    xpos, ypos = np.meshgrid(np.arange(nbins) * width, np.arange(nbins) * width)
    nonzero = histogram != 0
    # plot small contributions in units of 10^exponent W/m/K to keep tick labels short
    exponent = get_z_exponent(histogram)
    zunit = "W/m/K" if exponent == 0 else rf"$10^{{{exponent}}}$ W/m/K"
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.view_init(elev=elev, azim=azim)
    # the saved (tight) bbox always includes the full axes area, so enlarge the 3D box
    # to fill it instead of leaving empty space above the plot
    ax.set_box_aspect(None, zoom=1.15)
    if nonzero.any():
        ax.bar3d(
            xpos[nonzero],
            ypos[nonzero],
            np.zeros(nonzero.sum()),
            0.9 * width,
            0.9 * width,
            histogram[nonzero] / 10.0**exponent,
            color=COLORS[part] if color is None else color,
            edgecolor="none",
        )
    ax.set_xlim(0, nbins * width)
    ax.set_ylim(0, nbins * width)
    _set_3d_labels(
        ax, direction, zunit, xy_tick_pad=-2, z_tick_pad=1, xy_labelpad=-4, zlabelpad=0
    )
    return _finish_figure(fig, fig_path, fig_formats, show, return_fig, verbose)


def make_kappa_surface_plot(
    histogram,
    delta_freq,
    direction="xx",
    fig_path="kappa_surface.pdf",
    fig_formats=None,
    cmap=SURFACE_CMAP,
    gamma=0.6,
    upsample=8,
    min_fraction=2e-3,
    elev=25,
    azim=-120,
    show=False,
    return_fig=False,
    verbose=True,
):
    """
    Smooth 3D surface of a frequency-resolved kappa histogram, colored by height, after
    the style of Xia et al., Phys. Rev. X 10, 041029 (2020), Fig. 6(d,e).

    The histogram is placed at the bin centers and upsampled with cubic interpolation
    (negative overshoots are clipped to zero). Regions below min_fraction of the maximum
    are left empty so the floor shows where there is no contribution.

    Meant for the off-diagonal part. The diagonal part is nonzero only in the
    nu_i = nu_j bins, a ridge one bin wide that cubic interpolation rings around and
    that the masking cannot draw; use make_kappa_histogram_plot for it.

    Args:
        histogram (np.array[nbins, nbins]): contributions in W/m/K
            (see read_kappa_histogram)
        delta_freq (float): bin width in rad/ps
        direction (str): "xx", "yy", or "zz", used for the axis label
        fig_path (str | Path | None): path to save the figure. If None, don't save.
        fig_formats (str | iterable[str] | None): see make_kappa_histogram_plot
        cmap (str | Colormap): colormap for the surface height
        gamma (float): colors follow (height / max)**gamma, so gamma < 1 gives more of
            the surface the warmer colors of the upper colormap
        upsample (int): interpolation factor along each axis (1 for no interpolation)
        min_fraction (float): values below min_fraction * max are not drawn
        elev (float): elevation of the 3D view in degrees
        azim (float): azimuth of the 3D view in degrees
        show (bool): if True, call plt.show()
        return_fig (bool): if True, return the Figure instead of closing it
        verbose (bool): if True, print the save paths

    Returns:
        Figure if return_fig is True, otherwise None
    """
    _check_part_direction("od", direction)
    histogram = _check_histogram(histogram)
    nbins = histogram.shape[0]
    width = delta_freq / (2 * np.pi)  # rad/ps -> THz
    length = nbins * width

    exponent = get_z_exponent(histogram)
    zunit = "W/m/K" if exponent == 0 else rf"$10^{{{exponent}}}$ W/m/K"
    values = histogram / 10.0**exponent
    if upsample > 1:
        values = np.clip(upsample_array(values, upsample, order=3), 0, None)
    centers = np.linspace(0.5 * width, length - 0.5 * width, values.shape[0])
    xpos, ypos = np.meshgrid(centers, centers)  # x: mode j (columns), y: mode i (rows)
    zmax = values.max()
    ztop = 1.05 * zmax if zmax > 0 else 1.0

    fig = plt.figure(figsize=(6, 4.5))
    ax = fig.add_subplot(projection="3d")
    ax.view_init(elev=elev, azim=azim)
    ax.set_box_aspect((1, 1, 0.65))
    if zmax > 0:
        surface = np.where(values >= min_fraction * zmax, values, np.nan)
        # half-strength lighting shows the relief without washing out the colors
        facecolors = LightSource(azdeg=315, altdeg=45).shade(
            values,
            cmap=plt.get_cmap(cmap),
            norm=PowerNorm(gamma, vmin=0, vmax=zmax),
            blend_mode="soft",
            fraction=0.5,
        )
        ax.plot_surface(
            xpos,
            ypos,
            surface,
            facecolors=facecolors,
            rstride=1,
            cstride=1,
            linewidth=0,
            antialiased=False,
            shade=False,
            rasterized=True,  # keeps vector outputs small; axes and text stay vector
        )
    ax.set_xlim(0, length)
    ax.set_ylim(0, length)
    ax.set_zlim(0, ztop)

    # gray floor, no grid or walls, dashed bounding box
    ax.grid(False)
    ax.xaxis.set_pane_color((1, 1, 1, 0))
    ax.yaxis.set_pane_color((1, 1, 1, 0))
    ax.zaxis.set_pane_color((0.9, 0.9, 0.9, 1))
    corners = [(0, 0), (length, 0), (length, length), (0, length)]
    for (x0, y0), (x1, y1) in zip(corners, corners[1:] + corners[:1]):
        ax.plot([x0, x1], [y0, y1], [ztop, ztop], "k--", lw=0.6)
    # vertical edges, except at the corner nearest the viewer, which would cross the plot
    nearest = (
        0.0 if np.cos(np.radians(azim)) < 0 else length,
        0.0 if np.sin(np.radians(azim)) < 0 else length,
    )
    for x0, y0 in corners:
        if (x0, y0) != nearest:
            ax.plot([x0, x0], [y0, y0], [0, ztop], "k--", lw=0.6)

    _set_3d_labels(
        ax, direction, zunit, xy_tick_pad=-2, z_tick_pad=-2, xy_labelpad=-2, zlabelpad=-7
    )
    # crop to what is drawn: a tight bbox would keep the whole (mostly empty) axes area
    return _finish_figure(
        fig,
        fig_path,
        fig_formats,
        show,
        return_fig,
        verbose,
        bbox_inches=lambda: _get_content_bbox(fig, ax),
    )


def make_kappa_histogram_plots(
    filename_prefix,
    parts=("d", "od"),
    directions=("xx",),
    fig_formats=("pdf",),
    style="surface",
    verbose=True,
    **kwargs,
):
    """
    Read histograms written by UnifiedkappaManager and save a 3D plot of each to
    {filename_prefix}-{part}_{direction}.{fig_format} for each format in fig_formats.

    Args:
        filename_prefix (str | Path): prefix used when the histograms were written
        parts (iterable[str]): any of "d", "od"
        directions (iterable[str]): any of "xx", "yy", "zz"
        fig_formats (str | iterable[str]): figure file extensions, e.g. ("pdf", "png")
        style (str): "surface" (make_kappa_surface_plot) or "bar"
            (make_kappa_histogram_plot). The diagonal part ("d") is always plotted with
            bars, since it lies only on the nu_i = nu_j bins.
        verbose (bool): if True, print the save paths
        **kwargs: passed to the plotting function

    Returns:
        fig_paths (list[Path]): paths of the saved figures
    """
    if style not in PLOT_STYLES:
        raise ValueError(f"style must be one of {list(PLOT_STYLES)}, got {style!r}")
    fig_paths = []
    for part in parts:
        for direction in directions:
            histogram, delta_freq = read_kappa_histogram(filename_prefix, part, direction)
            # strip only ".txt": prefixes like "minikappa-300-2.0" contain dots
            histogram_path = get_histogram_path(filename_prefix, part, direction)
            fig_path = histogram_path.with_suffix("")
            plot_kwargs = dict(
                direction=direction,
                fig_path=fig_path,
                fig_formats=fig_formats,
                verbose=verbose,
                **kwargs,
            )
            if style == "surface" and part == "od":
                make_kappa_surface_plot(histogram, delta_freq, **plot_kwargs)
            else:
                make_kappa_histogram_plot(histogram, delta_freq, part=part, **plot_kwargs)
            fig_paths.extend(get_fig_paths(fig_path, fig_formats))
    return fig_paths


def get_z_exponent(histogram):
    """
    Power of ten to plot a histogram in: the exponent of its largest magnitude if that is
    below 0.01 W/m/K (so ticks read e.g. 0-5 instead of 0.000-0.005), otherwise 0.
    """
    max_value = np.max(np.abs(histogram), initial=0.0)
    if max_value == 0:
        return 0
    exponent = int(np.floor(np.log10(max_value)))
    return exponent if exponent <= -2 else 0


def get_fig_paths(fig_path, fig_formats=None):
    """
    Paths a figure is saved to.

    Args:
        fig_path (str | Path | None): figure path, with or without an extension
        fig_formats (str | iterable[str] | None): if None, [fig_path] as given.
            Otherwise one path per format: fig_path's extension is replaced if it is a
            figure format matplotlib knows (e.g. ".pdf"), and the format is appended
            otherwise (so "minikappa-300-2.0-d_xx" keeps its ".0-d_xx").

    Returns:
        fig_paths (list[Path])
    """
    if fig_path is None:
        return []
    fig_path = Path(fig_path)
    if fig_formats is None:
        return [fig_path]
    if isinstance(fig_formats, str):
        fig_formats = [fig_formats]
    supported = FigureCanvasBase.get_supported_filetypes()
    if fig_path.suffix[1:].lower() in supported:
        fig_path = fig_path.with_suffix("")
    fig_paths = []
    for fig_format in fig_formats:
        fig_format = fig_format.lstrip(".").lower()
        if fig_format not in supported:
            raise ValueError(
                f"Unsupported figure format {fig_format!r}; "
                f"use one of {sorted(supported)}"
            )
        fig_paths.append(fig_path.parent / f"{fig_path.name}.{fig_format}")
    return fig_paths


def _finish_figure(
    fig, fig_path, fig_formats, show, return_fig, verbose, bbox_inches="tight"
):
    """
    Save, show, and return or close a figure. bbox_inches may be a callable, which is
    only evaluated if the figure is saved.
    """
    fig_paths = get_fig_paths(fig_path, fig_formats)
    if fig_paths and callable(bbox_inches):
        bbox_inches = bbox_inches()
    for path in fig_paths:
        _save_figure(fig, path, bbox_inches=bbox_inches)
        if verbose:
            print(f"Saved figure to {path}")
    if show:
        plt.show()
    if return_fig:
        return fig
    plt.close(fig)
    return None


def _save_figure(fig, fig_path, bbox_inches="tight"):
    """
    Save without a creation date (and, for SVG, with fixed element ids), so identical
    plots give identical files.
    """
    no_date = {"pdf": {"CreationDate": None}, "svg": {"Date": None}}
    metadata = no_date.get(Path(fig_path).suffix[1:].lower())
    with plt.rc_context({"svg.hashsalt": "unifiedkappa"}):
        fig.savefig(fig_path, bbox_inches=bbox_inches, metadata=metadata)


def _get_content_bbox(fig, ax, pad=0.08):
    """
    Bounding box (inches) of the axis labels and ticks, lines, and collections of a 3D
    axes, padded by pad inches. Unlike bbox_inches="tight", it leaves out the empty part
    of the axes area around the 3D box.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = [axis.get_tightbbox(renderer) for axis in (ax.xaxis, ax.yaxis, ax.zaxis)]
    boxes += [artist.get_window_extent(renderer) for artist in ax.get_lines()]
    boxes += [artist.get_window_extent(renderer) for artist in ax.collections]
    boxes = [box for box in boxes if box is not None and box.width > 0 and box.height > 0]
    return Bbox.union(boxes).transformed(fig.dpi_scale_trans.inverted()).padded(pad)


def _set_3d_labels(ax, direction, zunit, xy_tick_pad, z_tick_pad, xy_labelpad, zlabelpad):
    """
    Axis labels, with tick labels and axis labels closer to the axes than the mplot3d
    defaults (which add a fixed 8 points to the tick pad and 21 points to the labelpad).
    3D tick labels are centered on their anchor, so wide labels (e.g. "0.05") need a
    larger pad than single digits.
    """
    ax.tick_params(axis="x", which="major", pad=xy_tick_pad)
    ax.tick_params(axis="y", which="major", pad=xy_tick_pad)
    ax.tick_params(axis="z", which="major", pad=z_tick_pad)
    ax.set_xlabel(r"$\nu_j$ (THz)", labelpad=xy_labelpad)
    ax.set_ylabel(r"$\nu_i$ (THz)", labelpad=xy_labelpad)
    ax.set_zlabel(rf"$\kappa_{{{direction}}}$ ({zunit})", labelpad=zlabelpad)


def _check_histogram(histogram):
    histogram = np.asarray(histogram, dtype=float)
    if histogram.ndim != 2 or histogram.shape[0] != histogram.shape[1]:
        raise ValueError(f"histogram must be square, got shape {histogram.shape}")
    return histogram


def _check_part_direction(part, direction):
    if part not in PARTS:
        raise ValueError(f"part must be one of {list(PARTS)}, got {part!r}")
    if direction not in DIRECTIONS:
        raise ValueError(
            f"direction must be one of {list(DIRECTIONS)}, got {direction!r}"
        )
