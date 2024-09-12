import numpy as np
import numpy.typing as npt

from typing import Any
from ase import Atoms


# left only for backward compatibility with old pickled dataframes
class TPAtomsDataContainer:

    def __init__(
        self,
        ase_atoms: Atoms,
        cutoff: float = 6.0,
        energy: npt.NDArray[np.float64] = None,
        forces: npt.NDArray[np.float64] = None,
        stress: npt.NDArray[np.float64] = None,
        atomic_nelec: npt.NDArray[np.float64] = None,
        total_nelec: float = None,
        mag_mom: npt.NDArray[np.float64] = None,
        atomic_chrg: npt.NDArray[np.float64] = None,
        total_chrg: npt.NDArray[np.float64] = None,
        atomic_dpl_mom: npt.NDArray[np.float64] = None,
        total_dpl_mom: npt.NDArray[np.float64] = None,
        neighborlist: Any = None,
        verbose: bool = True,
        struc_id: Any = None,
    ):
        pass
