# Copyright (c) Yi Xia and Dale Gaines II
# Distributed under the terms of the BSD 3-Clause License

"""Reading ShengBTE/FourPhonon inputs and outputs."""

import io
import re
from collections import defaultdict
from pathlib import Path

import f90nml
import numpy as np

# Fortran Ew.d drops the "E" when the exponent needs three digits:
# -1.66e-203 is written "-0.1663399480-202".
_MISSING_E = re.compile(rb"(?<=\d)([-+])(\d{3})(?=\s|$)", re.MULTILINE)


class ShengBTEOverflowError(ValueError):
    """ShengBTE output holds values >= 1e100 (or NaN/Inf in a kappa file).

    Subclasses ValueError so callers that already catch ValueError keep treating
    the run as failed.
    """


def load_bte(path: str | Path, require_finite: bool = False, **kwargs) -> np.ndarray:
    """Read a numeric ShengBTE/FourPhonon output file, like np.loadtxt.

    ShengBTE writes Fortran Ew.d fields, which drop the "E" when the exponent
    needs three digits: -1.66e-203 appears as "-0.1663399480-202". np.loadtxt
    cannot parse that. This reader distinguishes the two cases by the
    exponent's sign:

    - Negative (|x| < 1e-99): numerical zeros, e.g. symmetry-forbidden
      off-diagonal kappa. The "E" is restored and the value is parsed.
    - Positive (|x| >= 1e100): never physical in ShengBTE output; the run has
      diverged. Raises ShengBTEOverflowError.

    Args:
        path: File to read.
        require_finite: Also raise ShengBTEOverflowError on NaN/Inf. Use for
            kappa files only: healthy runs write NaN scattering rates for the
            three acoustic modes at Gamma (omega = 0) in BTE.w_final.
        **kwargs: Passed to np.loadtxt (usecols, ndmin, ...).

    Returns:
        The parsed array.

    Raises:
        ShengBTEOverflowError: Values >= 1e100, or NaN/Inf with require_finite.
            Subclasses ValueError, so callers that catch ValueError still treat
            the run as failed.

    Values like 1e98, written with a normal two-digit exponent, are not caught.
    """
    raw = Path(path).read_bytes()
    huge = [m for m in _MISSING_E.finditer(raw) if m.group(1) == b"+"]
    if huge:
        examples = ", ".join(_token_at(raw, m) for m in huge[:3])
        raise ShengBTEOverflowError(
            f"{path}: {len(huge)} values >= 1e100, e.g. {examples}"
        )
    data = np.loadtxt(io.BytesIO(_MISSING_E.sub(rb"E\1\2", raw)), **kwargs)
    if require_finite and not np.all(np.isfinite(data)):
        n_bad = np.sum(~np.isfinite(data))
        raise ShengBTEOverflowError(f"{path}: {n_bad} NaN/Inf values")
    return data


def _token_at(raw: bytes, match: re.Match) -> str:
    """Whole whitespace-delimited token ending at `match`, for error messages."""
    start = match.start()
    while start > 0 and not raw[start - 1 : start].isspace():
        start -= 1
    return raw[start : match.end()].decode()


class ShengBTEAnalyzer:
    """
    Simple class to extract information from ShengBTE/FourPhonon inputs and outputs
    """

    def __init__(
        self,
        workdir: str | Path,
        scattering_rate_cutoff: float = 0.0,
        kappa_precision: int | None = 5,
    ) -> None:
        """
        Args:
            workdir: Path to the ShengBTE/FourPhonon calculation directory.
            scattering_rate_cutoff: Rates at or below this value are replaced
                with 1e10 to prevent unphysically long lifetimes. Set to None
                to use raw data.
            kappa_precision: Decimal places to round every kappa tensor and
                scalar (including cumulative kappa). Set to None for raw data.
        """
        self.workdir = Path(workdir).resolve()
        self.scattering_rate_cutoff = scattering_rate_cutoff
        self.kappa_precision = kappa_precision

        self._check_complete()
        self._read_control()
        self._read_frequencies()
        self._read_dos()
        self._read_qpoints()
        self._read_group_velocities()
        self._read_phase_space()
        if not self.onlyharmonic:
            self._read_gruneisen()
            self._read_scattering_rates()
            self._read_kappa()
            self._read_unified_kappa()
            self._read_cumulative_kappas()

    def _check_complete(self) -> None:
        bte_out_path = self.workdir / "BTE.out"
        if not bte_out_path.exists():
            raise FileNotFoundError(f"No BTE.out file in {self.workdir}... exiting")
        with open(bte_out_path) as fr:
            bte_out = [line.strip() for line in fr.readlines()]
        complete = False
        goodlines = ["normal exit", "onlyharmonic=.true., stopping here"]
        for line in bte_out:
            if any(goodline in line for goodline in goodlines):
                complete = True
        if not complete:
            raise RuntimeError(f"BTE calculation failed in {self.workdir}... exiting")

    def _read_control(self) -> None:
        control_path = self.workdir / "CONTROL"
        raw_control_dict = f90nml.read(control_path).todict()
        control = {}
        control.update(raw_control_dict["allocations"])
        control.update(raw_control_dict["crystal"])
        control.update(raw_control_dict["parameters"])
        control.update(raw_control_dict["flags"])
        control.pop("_start_index")
        self._control = control

    @property
    def control(self) -> dict:
        return self._control

    @property
    def nelements(self) -> int:
        return self.control["nelements"]

    @property
    def natoms(self) -> int:
        return self.control["natoms"]

    @property
    def ngrid(self) -> np.ndarray:
        return np.asarray(self.control["ngrid"], dtype=int)

    @property
    def lfactor(self) -> float:
        return self.control.get("lfactor", 1.0)

    @property
    def lattice(self) -> np.ndarray:
        return np.asarray(self.control["lattvec"], dtype=float) * self.lfactor

    @property
    def volume(self) -> float:
        return np.abs(np.linalg.det(self.lattice))

    @property
    def elements(self) -> list[str]:
        return self.control["elements"]

    @property
    def positions(self) -> np.ndarray:
        return np.asarray(self.control["positions"], dtype=float)

    @property
    def scell(self) -> np.ndarray:
        return np.asarray(self.control["scell"], dtype=int)

    @property
    def scell_matrix(self) -> np.ndarray:
        return np.diag(self.scell)

    @property
    def temperatures(self) -> np.ndarray:
        if "t" in self.control:
            return np.asarray([self.control["t"]], dtype=int)
        else:
            t_min = self.control["t_min"]
            t_max = self.control["t_max"]
            t_step = self.control["t_step"]
            return np.arange(t_min, t_max + 1, t_step, dtype=int)

    @property
    def scalebroad(self) -> float:
        return self.control.get("scalebroad", 1.0)

    @property
    def maxiter(self) -> int:
        return self.control.get("maxiter", 1000)

    @property
    def num_sample_process_3ph_phase_space(self) -> int:
        return self.control.get("num_sample_process_3ph_phase_space", -1)

    @property
    def num_sample_process_3ph(self) -> int:
        return self.control.get("num_sample_process_3ph", -1)

    @property
    def num_sample_process_4ph_phase_space(self) -> int:
        return self.control.get("num_sample_process_4ph_phase_space", -1)

    @property
    def num_sample_process_4ph(self) -> int:
        return self.control.get("num_sample_process_4ph", -1)

    @property
    def convergence(self) -> bool:
        return self.control.get("convergence", True)

    @property
    def onlyharmonic(self) -> bool:
        return self.control.get("onlyharmonic", False)

    @property
    def four_phonon(self) -> bool:
        return self.control.get("four_phonon", False)

    @property
    def four_phonon_iteration(self) -> bool:
        return self.control.get("four_phonon_iteration", False)

    @property
    def nbands(self) -> int:
        return self.natoms * 3

    def _read_frequencies(self) -> None:
        omega_path = self.workdir / "BTE.omega"
        self._frequencies = load_bte(omega_path)

    @property
    def frequencies(self) -> np.ndarray:
        return self._frequencies

    def _read_dos(self) -> None:
        dos_path = self.workdir / "BTE.dos"
        self._dos = load_bte(dos_path, usecols=1)
        pdos_path = self.workdir / "BTE.pdos"
        self._pdos = load_bte(pdos_path)[:, 1:]

    @property
    def dos(self) -> np.ndarray:
        return self._dos

    @property
    def pdos(self) -> np.ndarray:
        return self._pdos

    def map_flat_quantity_to_q_nu(self, flat_quantity: np.ndarray) -> np.ndarray:
        """Reshape flat array of length nq*nbands to (nq, nbands)."""
        return flat_quantity.reshape((-1, self.nbands), order="F")

    def map_ibz_quantity_to_fbz(self, ibz_quantity: np.ndarray) -> np.ndarray:
        """Map an IBZ quantity of shape (nq_ibz, nu) to FBZ shape (nq_fbz, nu)."""
        fbz_quantity = np.zeros((self.nq_fbz, self.nbands), dtype=float)
        for i, idx in enumerate(self.idx_fbz_to_ibz):
            fbz_quantity[idx] = ibz_quantity[i]
        return fbz_quantity

    def _get_idx_fbz_to_ibz(self, qi_fbz_in_ibz: np.ndarray) -> list[np.ndarray]:
        """For each IBZ qpoint, find the corresponding FBZ qpoint indices.

        Args:
            qi_fbz_in_ibz: Array of IBZ indices, one per FBZ qpoint (length nq_fbz).

        Returns:
            List of arrays of FBZ indices, one per IBZ qpoint (length nq_ibz).
        """
        idx_fbz_to_ibz = []
        for i in range(1, self.nq + 1):
            idx_fbz_to_ibz.append(np.where(qi_fbz_in_ibz == i)[0])
        return idx_fbz_to_ibz

    def _read_qpoints(self) -> None:
        qpoints_path = self.workdir / "BTE.qpoints"
        qpoints_file = load_bte(qpoints_path)
        qpoints = qpoints_file[:, 3:]

        qpoints_full_path = self.workdir / "BTE.qpoints_full"
        qpoints_full_file = load_bte(qpoints_full_path)
        qi_fbz_in_ibz = qpoints_full_file[:, 1]
        qpoints_fbz = qpoints_full_file[:, 2:]

        self._qpoints = qpoints
        self._qpoints_fbz = qpoints_fbz
        self._idx_fbz_to_ibz = self._get_idx_fbz_to_ibz(qi_fbz_in_ibz)

    @property
    def qpoints(self) -> np.ndarray:
        return self._qpoints

    @property
    def nq(self) -> int:
        return len(self.qpoints)

    @property
    def qpoints_fbz(self) -> np.ndarray:
        return self._qpoints_fbz

    @property
    def nq_fbz(self) -> int:
        return len(self.qpoints_fbz)

    @property
    def qpoint_degeneracy(self) -> np.ndarray:
        return np.asarray([len(qpoints_fbz) for qpoints_fbz in self.idx_fbz_to_ibz])

    @property
    def qpoint_weights(self) -> np.ndarray:
        return self.qpoint_degeneracy / self.nq_fbz

    @property
    def idx_fbz_to_ibz(self) -> list[np.ndarray]:
        return self._idx_fbz_to_ibz

    def _read_group_velocities(self) -> None:
        group_velocity_path = self.workdir / "BTE.v"
        group_velocity = load_bte(group_velocity_path)
        group_velocity = group_velocity.reshape(
            (self.nq, self.nbands, 3),
            order="F",
        )
        self._group_velocity = group_velocity
        group_velocity_fbz_path = self.workdir / "BTE.v_full"
        group_velocity_fbz = load_bte(group_velocity_fbz_path)
        group_velocity_fbz = group_velocity_fbz.reshape(
            (self.nq_fbz, self.nbands, 3),
            order="F",
        )
        self._group_velocity_fbz = group_velocity_fbz

    @property
    def group_velocity(self) -> np.ndarray:
        return self._group_velocity

    @property
    def group_velocity_fbz(self) -> np.ndarray:
        return self._group_velocity_fbz

    def _read_phase_space(self) -> None:
        p3_path = self.workdir / "BTE.P3"
        self._p3 = load_bte(p3_path)
        p4_path = self.workdir / "BTE.P4"
        if p4_path.exists():
            self._p4 = load_bte(p4_path)

    @property
    def p3(self) -> np.ndarray:
        return self._p3

    @property
    def p4(self) -> np.ndarray:
        return self._p4

    @property
    def p3_total(self) -> float:
        return np.sum(self.qpoint_degeneracy.reshape(-1, 1) * self.p3)

    @property
    def p4_total(self) -> float:
        return np.sum(self.qpoint_degeneracy.reshape(-1, 1) * self.p4)

    def _read_gruneisen(self) -> None:
        gruneisen_path = self.workdir / "BTE.gruneisen"
        self._gruneisen = load_bte(gruneisen_path)

    @property
    def gruneisen(self) -> np.ndarray:
        return self._gruneisen

    @property
    def temperature_dirs(self) -> list[Path]:
        """Search the BTE directory for all temperature directories matching T*K."""
        return [
            path
            for path in sorted(
                self.workdir.glob("T*K"), key=lambda d: int(d.name.strip("TK"))
            )
            if path.is_dir()
        ]

    def _read_scattering_rates(self) -> None:
        rates_map = {
            "rates3": "BTE.w_3ph",
            "rates4": "BTE.w_4ph",
            "rates_rta": "BTE.w",
            "rates": "BTE.w_final",
        }
        for temperature_dir in self.temperature_dirs:
            temperature = temperature_dir.name.strip("TK")
            for rate_type, rate_filename in rates_map.items():
                rates_file = temperature_dir / rate_filename
                if not rates_file.exists():
                    continue
                rates = load_bte(rates_file, usecols=1)
                rates = self.map_flat_quantity_to_q_nu(rates)
                if self.scattering_rate_cutoff is not None:
                    rates = np.nan_to_num(rates, nan=0.0)
                    rates = np.where(
                        rates > self.scattering_rate_cutoff,
                        rates,
                        1e10,
                    )
                hidden_rate_name = f"_{rate_type}"
                current_rate_attr = getattr(self, hidden_rate_name, None)
                if current_rate_attr is None:
                    current_rate_attr = {}
                    setattr(self, hidden_rate_name, current_rate_attr)
                current_rate_attr[temperature] = rates

    @property
    def rates(self) -> dict[str, np.ndarray]:
        return self._rates

    @property
    def rates_rta(self) -> dict[str, np.ndarray]:
        return self._rates_rta

    @property
    def rates3(self) -> dict[str, np.ndarray]:
        return self._rates3

    @property
    def rates4(self) -> dict[str, np.ndarray]:
        return self._rates4

    @property
    def rates_fbz(self) -> dict[str, np.ndarray]:
        rates_fbz = {}
        for temperature, rates_T in self.rates.items():
            rates_fbz[temperature] = self.map_ibz_quantity_to_fbz(rates_T)
        return rates_fbz

    @property
    def rates_rta_fbz(self) -> dict[str, np.ndarray]:
        rates_rta_fbz = {}
        for temperature, rates_T in self.rates_rta.items():
            rates_rta_fbz[temperature] = self.map_ibz_quantity_to_fbz(rates_T)
        return rates_rta_fbz

    @property
    def rates3_fbz(self) -> dict[str, np.ndarray]:
        rates3_fbz = {}
        for temperature, rates_T in self.rates3.items():
            rates3_fbz[temperature] = self.map_ibz_quantity_to_fbz(rates_T)
        return rates3_fbz

    @property
    def rates4_fbz(self) -> dict[str, np.ndarray]:
        rates4_fbz = {}
        for temperature, rates_T in self.rates4.items():
            rates4_fbz[temperature] = self.map_ibz_quantity_to_fbz(rates_T)
        return rates4_fbz

    def _round_kappa(self, kappa: np.ndarray | float) -> np.ndarray | float:
        """Round a kappa tensor or scalar to kappa_precision (None: unchanged)."""
        if self.kappa_precision is None:
            return kappa
        return np.round(kappa, decimals=self.kappa_precision)

    def get_kappa_as_scalar(self, kappa_tensor: np.ndarray) -> float:
        """Mean of the diagonal of a kappa tensor (length 9 or 3x3), rounded."""
        if len(kappa_tensor) == 9:
            kappa_tensor = kappa_tensor.reshape(3, 3)
        return self._round_kappa(np.mean(np.diag(kappa_tensor)))

    def _read_kappa(self) -> None:
        self._kappa = {}
        self._kappa_tensor = {}
        self._kappa_rta = {}
        self._kappa_rta_tensor = {}
        for temperature_dir in self.temperature_dirs:
            temperature = temperature_dir.name.strip("TK")
            kappa_tensor_path = temperature_dir / "BTE.kappa_tensor"
            kappa_tensor_file = np.atleast_2d(
                load_bte(kappa_tensor_path, require_finite=True)
            )
            # Last row = converged (iterative) kappa
            kappa_tensor = kappa_tensor_file[-1, 1:]
            kappa = self.get_kappa_as_scalar(kappa_tensor)
            # First row (iteration 0) = RTA kappa
            kappa_rta_tensor = kappa_tensor_file[0, 1:]
            kappa_rta = self.get_kappa_as_scalar(kappa_rta_tensor)
            self._kappa[temperature] = kappa
            self._kappa_tensor[temperature] = self._round_kappa(kappa_tensor)
            self._kappa_rta[temperature] = kappa_rta
            self._kappa_rta_tensor[temperature] = self._round_kappa(kappa_rta_tensor)

    @property
    def kappa(self) -> dict[str, float]:
        return self._kappa

    @property
    def kappa_tensor(self) -> dict[str, np.ndarray]:
        return self._kappa_tensor

    @property
    def kappa_rta(self) -> dict[str, float]:
        return self._kappa_rta

    @property
    def kappa_rta_tensor(self) -> dict[str, np.ndarray]:
        return self._kappa_rta_tensor

    def _read_unified_kappa(self) -> None:
        unified_results = defaultdict(dict)
        for temperature_dir in self.temperature_dirs:
            temperature = temperature_dir.name.strip("TK")
            unified_kappa_path = temperature_dir / "unifiedkappa.dat"
            if not unified_kappa_path.exists():
                continue
            unified_kappa_tensor_file = load_bte(unified_kappa_path, require_finite=True)
            ukappa_names = ["unified_kappa_d", "unified_kappa_od", "unified_kappa"]
            for i, ukappa_name in enumerate(ukappa_names):
                unified_results[ukappa_name][temperature] = unified_kappa_tensor_file[i]

        for ukappa_name, temperature_dict in unified_results.items():
            for temperature, ukappa_tensor in temperature_dict.items():
                ukappa_scalar = self.get_kappa_as_scalar(ukappa_tensor)
                ukappa_tensor = self._round_kappa(ukappa_tensor)

                ukappa_attr_name = f"_{ukappa_name}"
                current_ukappa_attr = getattr(self, ukappa_attr_name, None)
                if current_ukappa_attr is None:
                    current_ukappa_attr = {}
                    setattr(self, ukappa_attr_name, current_ukappa_attr)
                current_ukappa_attr[temperature] = ukappa_scalar
                ukappa_tensor_attr_name = f"_{ukappa_name}_tensor"
                current_ukappa_tensor_attr = getattr(self, ukappa_tensor_attr_name, None)
                if current_ukappa_tensor_attr is None:
                    current_ukappa_tensor_attr = {}
                    setattr(self, ukappa_tensor_attr_name, current_ukappa_tensor_attr)
                current_ukappa_tensor_attr[temperature] = ukappa_tensor

    @property
    def unified_kappa(self) -> dict[str, float]:
        return self._unified_kappa

    @property
    def unified_kappa_D(self) -> dict[str, float]:
        return self._unified_kappa_d

    @property
    def unified_kappa_OD(self) -> dict[str, float]:
        return self._unified_kappa_od

    @property
    def unified_kappa_tensor(self) -> dict[str, np.ndarray]:
        return self._unified_kappa_tensor

    @property
    def unified_kappa_D_tensor(self) -> dict[str, np.ndarray]:
        return self._unified_kappa_d_tensor

    @property
    def unified_kappa_OD_tensor(self) -> dict[str, np.ndarray]:
        return self._unified_kappa_od_tensor

    def _read_cumulative_kappas(self) -> None:
        ckappa_map = {
            "ckappa_tensor_vs_mfp": {
                "fname": "BTE.cumulative_kappa_tensor",
                "xname": "mfp",
            },
            "ckappa_tensor_vs_omega": {
                "fname": "BTE.cumulative_kappaVsOmega_tensor",
                "xname": "omega",
            },
        }
        for temperature_dir in self.temperature_dirs:
            temperature = temperature_dir.name.strip("TK")
            for ckappa_name, ckappa_info in ckappa_map.items():
                ckappa_file = temperature_dir / ckappa_info["fname"]
                ckappa_data = load_bte(ckappa_file, require_finite=True)

                hidden_ckappa_name = f"_{ckappa_name}"
                current_ckappa_attr = getattr(self, hidden_ckappa_name, None)
                if current_ckappa_attr is None:
                    current_ckappa_attr = {}
                    setattr(self, hidden_ckappa_name, current_ckappa_attr)
                current_ckappa_attr[temperature] = {
                    ckappa_info["xname"]: ckappa_data[:, 0],
                    "ckappa": self._round_kappa(ckappa_data[:, 1:]),
                }

    @property
    def ckappa_vs_omega(self) -> dict[str, dict]:
        ckappa_vs_omega = {}
        for temperature, ckappa_dict in self.ckappa_tensor_vs_omega.items():
            ckappa_vs_omega[temperature] = {
                "omega": ckappa_dict["omega"],
                "ckappa": np.apply_along_axis(
                    self.get_kappa_as_scalar,
                    1,
                    ckappa_dict["ckappa"],
                ),
            }
        return ckappa_vs_omega

    @property
    def ckappa_tensor_vs_omega(self) -> dict[str, dict]:
        return self._ckappa_tensor_vs_omega

    @property
    def ckappa_vs_mfp(self) -> dict[str, dict]:
        ckappa_vs_mfp = {}
        for temperature, ckappa_dict in self.ckappa_tensor_vs_mfp.items():
            ckappa_vs_mfp[temperature] = {
                "mfp": ckappa_dict["mfp"],
                "ckappa": np.apply_along_axis(
                    self.get_kappa_as_scalar,
                    1,
                    ckappa_dict["ckappa"],
                ),
            }
        return ckappa_vs_mfp

    @property
    def ckappa_tensor_vs_mfp(self) -> dict[str, dict]:
        return self._ckappa_tensor_vs_mfp


def unwrap_qpoints(qpoints: np.ndarray) -> np.ndarray:
    """Convert fractional qpoint coordinates to ShengBTE format (no negative values)."""
    n_qpoints = len(qpoints)
    qpoints = qpoints.flatten()
    qpoints = np.where(qpoints >= 0, qpoints, 1 - np.abs(qpoints))
    qpoints = qpoints.reshape(n_qpoints, -1)
    return qpoints


def wrap_qpoints(qpoints: np.ndarray) -> np.ndarray:
    """Convert fractional qpoint coordinates to phonopy format (minimise absolute value).

    Note: edge cases exist when coordinates are 0.5 or -0.5 — phonopy uses both
    signs for equivalent q-points and this function does not resolve that ambiguity.
    """
    n_qpoints = len(qpoints)
    qpoints = qpoints.flatten()
    qpoints = np.where(qpoints <= 0.5, qpoints, -1 + qpoints)
    qpoints = qpoints.reshape(n_qpoints, -1)
    return qpoints


if __name__ == "__main__":
    sbte = ShengBTEAnalyzer(".")
    print("qpoints analysis")
    print(f"{sbte.nq=} {sbte.qpoints.shape}")
    print(f"{sbte.qpoints=}")
    print(f"{sbte.nq_fbz=} {sbte.qpoints_fbz.shape}")
    print(f"{sbte.qpoints_fbz=}")

    print("Scattering rate analysis")
    for temperature in sbte.temperatures:
        print(f"{temperature}")
        print(f"{sbte.rates[f'{temperature}'].shape}")
        print(f'{sbte.rates[f"{temperature}"]=}')
        print(f"{sbte.rates_fbz[f'{temperature}'].shape}")
        print(f'{sbte.rates_fbz[f"{temperature}"]=}')

    print("Kappa analysis")
    for temperature in sbte.temperatures:
        print(f"{temperature}")
        print(f"{sbte.kappa[f'{temperature}']}")
        print(f"{sbte.kappa_tensor[f'{temperature}'].shape}")
        print(f"{sbte.kappa_tensor[f'{temperature}']}")
