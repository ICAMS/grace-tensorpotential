from abc import ABC, abstractmethod
import logging
from typing import Iterable, List, Mapping, Optional, Union, Iterator, NamedTuple
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


LOG_FMT = "%(asctime)s %(levelname).1s - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FMT, datefmt="%Y/%m/%d %H:%M:%S")
log = logging.getLogger()

DatasetContainer = Union[Mapping[str, pd.DataFrame], Iterable[pd.DataFrame]]


class UnitTransformTuple(NamedTuple):
    """
    Describes unit transformation for a plot target.

    If provided, label and units are included in the x-axis title.

    Attributes:
        units: Display units (e.g. 'GPa', 'eV/atom') shown in axis labels.
        label: Axis label (optional). Supports LaTeX formatting.
        factor: Multiplicative factor to convert raw data into target units.
    """

    units: str
    label: str = ""
    factor: float = 1


DEFAULT_PLOT_TARGETS = ("energy_corrected_per_atom", "forces", "stress")

# E: eV/atom, F: eV/A, stress: GPa
DEFAULT_UNIT_TRANSFORM = {
    "energy_corrected_per_atom": UnitTransformTuple("eV/atom", r"$E$"),
    "forces": UnitTransformTuple("eV/A", r"$F$"),
    "stress": UnitTransformTuple("GPa", r"$\sigma$", 160.2176621),
}


class IDatasetPlotter(ABC):
    @classmethod
    @abstractmethod
    def plot(
        cls,
        datasets: DatasetContainer,
        output_dir: Union[Path, str],
        plot_targets: Optional[Iterable[str]] = DEFAULT_PLOT_TARGETS,
        xscale: str = "linear",
        yscale: str = "linear",
        bins: Union[int, Iterable[int]] = 25,
        alpha_decay: float = 0.66,
        units_transform: Optional[Mapping[str, UnitTransformTuple]] = None,
        **kwargs,
    ) -> None:
        pass


class DatasetHistPlotter(IDatasetPlotter):
    """
    Histogram plotter for collections of datasets in pandas DataFrames with per-target unit transformation.

    Supports plotting multiple datasets (mapping or iterable of DataFrames),
    applying unit conversions, and generating per-target histograms.

    Data for each target is plotted across datasets on a shared histogram (shared bins).

    Usage:
        DatasetHistPlotter.plot(datasets, output_dir, ...)
    """

    @classmethod
    def plot(
        cls,
        datasets: DatasetContainer,
        output_dir: Union[Path, str],
        plot_targets: Optional[Iterable[str]] = DEFAULT_PLOT_TARGETS,
        xscale: str = "linear",
        yscale: str = "log",
        bins: Union[int, Iterable[int]] = 100,
        alpha_decay: float = 0.66,
        units_transform: Optional[Mapping[str, UnitTransformTuple]] = None,
        **kwargs,
    ) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for target in plot_targets:
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            plot_data = DatasetHistPlotter._get_target_data(datasets, target)
            if not plot_data:
                log.warning(
                    f'DatasetHistPlotter: found no data in any dataset for target "{target}", skipping'
                )
                continue

            units_factor = 1
            if units_transform is not None and target in units_transform:
                units_factor = units_transform[target].factor

            bins_ = cls._bins(plot_data, bins) if isinstance(bins, int) else bins

            alpha = 1.0
            for i, (ds_lbl, ds) in enumerate(
                DatasetHistPlotter._iter_datasets(datasets)
            ):
                try:
                    v = plot_data[i] * units_factor
                    ax.hist(v, bins=bins_, alpha=alpha, label=ds_lbl, **kwargs)
                    ax.set_title(f"{target.replace('_', ' ').title()} Distribution")
                    ax.set_xlabel(cls._make_x_lbl(target, units_transform))
                    ax.set_ylabel("Count")
                    ax.set_xscale(xscale)
                    ax.set_yscale(yscale)
                except Exception as e:
                    log.warning(
                        f'DatasetHistPlotter: Failed to plot distribution for target "{target}" for dataset "{ds_lbl}", skipping'
                    )
                alpha = cls._decrease_alpha(alpha, alpha_decay)

            plt.tight_layout()
            plt.legend()
            plt.savefig(output_dir / f"{target}.png")
            plt.close()

    @staticmethod
    def _bins(vs: Iterable[np.ndarray], n_bins: int) -> np.ndarray:
        x_min = np.min([np.min(v) for v in vs])
        x_max = np.max([np.max(v) for v in vs])
        return np.linspace(x_min, x_max, n_bins)

    @staticmethod
    def _get_target_data(datasets: DatasetContainer, target: str) -> List[np.ndarray]:
        data = []
        for ds_lbl, ds in DatasetHistPlotter._iter_datasets(datasets):
            try:
                data.append(ds[target].to_numpy())
            except KeyError:
                log.warning(
                    f'DatasetHistPlotter: Failed to get data for target "{target}" for dataset "{ds_lbl}", skipping'
                )

        return [DatasetHistPlotter._stack_v(v) for v in data]

    @staticmethod
    def _iter_datasets(
        datasets: DatasetContainer,
    ) -> Iterator[tuple[Union[str, int], pd.DataFrame]]:
        if isinstance(datasets, Mapping):
            return ((k, v) for k, v in datasets.items())
        else:
            return ((i, v) for i, v in enumerate(datasets))

    @staticmethod
    def _stack_v(v: np.ndarray) -> np.ndarray:
        if v.dtype == object:
            return np.vstack(v).flatten()
        else:
            return v.flatten()

    @staticmethod
    def _decrease_alpha(alpha: float, factor: float = 0.66):
        return alpha * factor

    @staticmethod
    def _make_x_lbl(
        target: str, units_transform: Optional[Mapping[str, UnitTransformTuple]] = None
    ) -> str:
        if not units_transform or target not in units_transform:
            return target

        suffix = units_transform[target].units
        suffix = f", {suffix}" if suffix else ""

        label = (
            units_transform[target].label if units_transform[target].label else target
        )

        return label + suffix
