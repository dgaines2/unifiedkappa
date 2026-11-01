from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.colors import to_rgba

from unifiedkappa import UnifiedkappaManager
from unifiedkappa.plotting import (
    COLORS,
    get_fig_paths,
    get_histogram_path,
    make_kappa_histogram_plot,
    make_kappa_histogram_plots,
    read_kappa_histogram,
)

NBINS = 8


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


@pytest.fixture
def histogram_files(nacl_phonon, tmp_path):
    """Histograms written by calculate_unified_kappa on synthetic inputs."""
    rng = np.random.default_rng(1)
    nqpt, nband = 10, 6
    freqs = rng.uniform(1.0, 40.0, size=(nqpt, nband))
    Gamma = rng.uniform(0.1, 5.0, size=(nqpt, nband))
    v = rng.normal(size=(nqpt, nband, nband, 3)) + 1j * rng.normal(
        size=(nqpt, nband, nband, 3)
    )
    gvfull = (v + v.conj().transpose(0, 2, 1, 3)) / 2
    manager = UnifiedkappaManager(
        nacl_phonon, symprec=None, save_histogram=True, n_histogram_bins=NBINS
    )
    prefix = tmp_path / "unifiedkappa"
    kappaD, kappaOD, _ = manager.calculate_unified_kappa(
        freqs, Gamma, gvfull, filename_prefix=str(prefix)
    )
    delta_freq = np.max(freqs + 0.1) / NBINS
    return prefix, {"d": kappaD, "od": kappaOD}, delta_freq


def test_get_histogram_path():
    path = get_histogram_path("T300K/unifiedkappa", "od", "yy")
    assert str(path) == "T300K/unifiedkappa-od_yy.txt"
    with pytest.raises(ValueError, match="part"):
        get_histogram_path("x", "offdiag", "xx")
    with pytest.raises(ValueError, match="direction"):
        get_histogram_path("x", "d", "xy")


@pytest.mark.parametrize("part", ["d", "od"])
@pytest.mark.parametrize("direction", ["xx", "yy", "zz"])
def test_read_kappa_histogram(histogram_files, part, direction):
    prefix, kappa, expected_delta_freq = histogram_files
    histogram, delta_freq = read_kappa_histogram(prefix, part, direction)
    assert histogram.shape == (NBINS, NBINS)
    assert delta_freq == pytest.approx(expected_delta_freq, rel=1e-9)
    i = "xyz".index(direction[0])
    assert histogram.sum() == pytest.approx(kappa[part][i, i], rel=1e-8)


def test_read_kappa_histogram_without_header(tmp_path):
    np.savetxt(tmp_path / "old-od_xx.txt", np.eye(3))
    with pytest.raises(ValueError, match="no delta_freq header"):
        read_kappa_histogram(tmp_path / "old")
    histogram, delta_freq = read_kappa_histogram(tmp_path / "old", delta_freq=0.5)
    np.testing.assert_array_equal(histogram, np.eye(3))
    assert delta_freq == 0.5


def test_make_kappa_histogram_plot(tmp_path):
    histogram = np.arange(16, dtype=float).reshape(4, 4)
    delta_freq = 2 * np.pi * 0.5  # 0.5 THz bins
    fig_path = tmp_path / "hist.png"
    fig = make_kappa_histogram_plot(
        histogram,
        delta_freq,
        part="d",
        direction="yy",
        fig_path=fig_path,
        return_fig=True,
        verbose=False,
    )
    assert fig_path.stat().st_size > 0
    (ax,) = fig.axes
    assert ax.name == "3d"
    np.testing.assert_allclose(ax.get_xlim(), [0, 2.0])
    np.testing.assert_allclose(ax.get_ylim(), [0, 2.0])
    assert ax.get_zlim()[1] >= histogram.max()
    assert "THz" in ax.get_xlabel() and "THz" in ax.get_ylabel()
    assert ax.get_zlabel() == r"$\kappa_{yy}$ (W/m/K)"
    assert ax.get_title() == ""
    (bars,) = ax.collections
    # bar3d shades each face by scaling the base color
    for face in bars.get_facecolor():
        rgb, base = face[:3], np.array(to_rgba(COLORS["d"])[:3])
        np.testing.assert_allclose(rgb / rgb.max(), base / base.max(), atol=1e-6)


def test_make_kappa_histogram_plot_skips_empty_bins():
    histogram = np.zeros((5, 5))
    histogram[1, 2] = histogram[2, 1] = 0.5
    fig = make_kappa_histogram_plot(
        histogram, 1.0, fig_path=None, return_fig=True, verbose=False
    )
    (bars,) = fig.axes[0].collections
    assert len(bars.get_facecolor()) == 2 * 6  # two bars, six faces each

    empty = make_kappa_histogram_plot(
        np.zeros((5, 5)), 1.0, fig_path=None, return_fig=True, verbose=False
    )
    assert len(empty.axes[0].collections) == 0


def test_make_kappa_histogram_plot_closes_figure(tmp_path):
    result = make_kappa_histogram_plot(
        np.ones((3, 3)), 1.0, fig_path=tmp_path / "hist.pdf", verbose=False
    )
    assert result is None
    assert plt.get_fignums() == []


def test_make_kappa_histogram_plot_rejects_non_square():
    with pytest.raises(ValueError, match="square"):
        make_kappa_histogram_plot(np.ones((3, 4)), 1.0, fig_path=None, verbose=False)


def test_make_kappa_histogram_plots(histogram_files):
    prefix, _, _ = histogram_files
    fig_paths = make_kappa_histogram_plots(
        prefix, directions=("xx", "zz"), fig_formats=("pdf", "png"), verbose=False
    )
    expected = [
        prefix.parent / f"unifiedkappa-{part}_{direction}.{fig_format}"
        for part in ("d", "od")
        for direction in ("xx", "zz")
        for fig_format in ("pdf", "png")
    ]
    assert fig_paths == expected
    assert all(path.stat().st_size > 0 for path in fig_paths)
    assert plt.get_fignums() == []


def test_make_kappa_histogram_plots_dotted_prefix(tmp_path):
    """Prefixes like minikappa-300-2.0 contain dots that must not be treated as
    extensions."""
    prefix = tmp_path / "minikappa-300-2.0"
    np.savetxt(f"{prefix}-od_xx.txt", np.ones((3, 3)), header="delta_freq = 1.0 rad/ps")
    fig_paths = make_kappa_histogram_plots(
        prefix, parts=("od",), fig_formats="png", verbose=False
    )
    assert fig_paths == [tmp_path / "minikappa-300-2.0-od_xx.png"]
    assert fig_paths[0].exists()


@pytest.mark.parametrize(
    "fig_path, fig_formats, expected",
    [
        (None, ("pdf", "png"), []),
        ("a/hist.pdf", None, ["a/hist.pdf"]),
        ("a/hist.pdf", ("pdf", "png"), ["a/hist.pdf", "a/hist.png"]),
        ("a/hist.PDF", "png", ["a/hist.png"]),
        ("a/hist", [".svg"], ["a/hist.svg"]),
        ("a/minikappa-300-2.0-d_xx", ("pdf",), ["a/minikappa-300-2.0-d_xx.pdf"]),
    ],
)
def test_get_fig_paths(fig_path, fig_formats, expected):
    assert get_fig_paths(fig_path, fig_formats) == [Path(path) for path in expected]


def test_get_fig_paths_unsupported_format():
    with pytest.raises(ValueError, match="Unsupported figure format 'docx'"):
        get_fig_paths("hist", ("pdf", "docx"))


def test_make_kappa_histogram_plot_multiple_formats(tmp_path):
    make_kappa_histogram_plot(
        np.ones((3, 3)),
        1.0,
        fig_path=tmp_path / "hist.pdf",
        fig_formats=("pdf", "png", "svg"),
        verbose=False,
    )
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "hist.pdf",
        "hist.png",
        "hist.svg",
    ]
    assert (tmp_path / "hist.png").read_bytes().startswith(b"\x89PNG")


@pytest.mark.parametrize("fig_format", ["pdf", "png", "svg"])
def test_saved_figures_are_reproducible(tmp_path, fig_format):
    """No creation date in the files, so identical plots give identical files."""
    contents = []
    for name in ["first", "second"]:
        make_kappa_histogram_plot(
            np.arange(9.0).reshape(3, 3),
            1.0,
            fig_path=tmp_path / name,
            fig_formats=fig_format,
            verbose=False,
        )
        contents.append((tmp_path / f"{name}.{fig_format}").read_bytes())
    assert contents[0] == contents[1]
