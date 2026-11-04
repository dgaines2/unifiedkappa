"""load_bte: ShengBTE output with three-digit exponents, NaN and Inf.

Fixtures in tests/data/ are unmodified ShengBTE output from htphonons:
  - underflow.BTE.cumulative_kappaVsOmega_tensor: icsd_simple_binaries/2012, LTC_3,
    T600K. A healthy run; one symmetry-forbidden off-diagonal entry underflowed to
    "-0.1663399480-202" (row 9, column 9).
  - overflow.BTE.kappa_tensor: icsd_complex_binaries/14331, LTC_3, T300K, from before
    the POSCAR-prim wrap fix. A diverged run: entries like "-0.1732710499+100".
  - nan.BTE.cumulative_kappa_scalar: same run, a kappa file that is NaN with no
    three-digit exponents.
"""

from pathlib import Path

import numpy as np
import pytest

from unifiedkappa import ShengBTEOverflowError, load_bte

DATA_DIR = Path(__file__).resolve().parent / "data"


def test_underflow_parsed():
    data = load_bte(DATA_DIR / "underflow.BTE.cumulative_kappaVsOmega_tensor")
    assert data.shape == (100, 10)
    assert data[8, 8] == pytest.approx(-1.663399480e-203, rel=1e-9)
    # np.loadtxt is what failed on this file
    with pytest.raises(ValueError):
        np.loadtxt(DATA_DIR / "underflow.BTE.cumulative_kappaVsOmega_tensor")


def test_underflow_passes_require_finite():
    data = load_bte(
        DATA_DIR / "underflow.BTE.cumulative_kappaVsOmega_tensor", require_finite=True
    )
    assert np.all(np.isfinite(data))


def test_overflow_raises():
    with pytest.raises(ShengBTEOverflowError, match=r"554 values >= 1e100") as excinfo:
        load_bte(DATA_DIR / "overflow.BTE.kappa_tensor")
    assert "-0.1732710499+100" in str(excinfo.value)
    # existing callers catch ValueError
    assert isinstance(excinfo.value, ValueError)


def test_nan_allowed_by_default():
    data = load_bte(DATA_DIR / "nan.BTE.cumulative_kappa_scalar")
    assert np.isnan(data[:, 1]).all()


def test_nan_raises_with_require_finite():
    with pytest.raises(ShengBTEOverflowError, match="NaN/Inf"):
        load_bte(DATA_DIR / "nan.BTE.cumulative_kappa_scalar", require_finite=True)


def test_gamma_nan_rates_allowed(shengbte_example_dir):
    """Healthy runs have NaN rates for the acoustic modes at Gamma."""
    rates = load_bte(shengbte_example_dir / "T300K" / "BTE.w_final", usecols=1)
    assert np.isnan(rates).sum() > 0


def test_matches_loadtxt_on_normal_file(shengbte_example_dir):
    path = shengbte_example_dir / "T300K" / "BTE.kappa_tensor"
    np.testing.assert_array_equal(load_bte(path), np.loadtxt(path))


def test_kwargs_passed_through(shengbte_example_dir):
    path = shengbte_example_dir / "BTE.dos"
    np.testing.assert_array_equal(load_bte(path, usecols=1), np.loadtxt(path, usecols=1))


@pytest.mark.parametrize(
    "text, expected",
    [
        ("-0.1663399480-202", -1.663399480e-203),  # missing E restored
        ("0.12E-02", 0.12e-02),  # normal exponent untouched
        ("-0.12E-101", -0.12e-101),  # three-digit exponent with E untouched
        ("  -5", -5.0),  # leading sign is not an exponent
        ("170", 170.0),
    ],
)
def test_exponent_repair(tmp_path, text, expected):
    path = tmp_path / "f"
    path.write_text(text + "\n")
    assert load_bte(path) == pytest.approx(expected)
