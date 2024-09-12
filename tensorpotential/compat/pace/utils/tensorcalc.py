import numpy as np
import tensorflow as tf

from ase.calculators.calculator import Calculator, all_changes

from tensorpotential.compat.pace.utils.neighborlist import PrimitiveNeighborListWrapper
from tensorpotential.compat.pace.utils.utilities import input2eval


class TensorCalc(Calculator):
    """
    TensorPotential ASE calculator
    """

    implemented_properties = ["energy", "forces", "stress"]

    def __init__(self, **kwargs):
        Calculator.__init__(self, **kwargs)
        self.nl = None
        self.skin = 0.0
        try:
            if self.parameters.reset_nl is not None:
                self.reset_nl = self.parameters.reset_nl
            else:
                self.reset_nl = False
        except:
            self.reset_nl = False  # Set to False for MD simulations

        self.model = tf.saved_model.load(self.parameters.model_location)

    def get_data(self, atoms):
        if self.reset_nl:
            self.nl = PrimitiveNeighborListWrapper(
                cutoffs=[self.parameters.cutoff * 0.5] * len(atoms),
                skin=self.skin,
                self_interaction=False,
                bothways=True,
                use_scaled_positions=False,
            )
            self.nl.update(atoms.get_pbc(), atoms.get_cell(), atoms.get_positions())
        else:
            if self.nl is None:
                self.nl = PrimitiveNeighborListWrapper(
                    cutoffs=[self.parameters.cutoff * 0.5] * len(atoms),
                    skin=self.skin,
                    self_interaction=False,
                    bothways=True,
                )  # , use_scaled_positions=False)
                self.nl.update(atoms.get_pbc(), atoms.get_cell(), atoms.get_positions())
            else:
                self.nl.update(atoms.get_pbc(), atoms.get_cell(), atoms.get_positions())

        self.data = {
            "ind_i": [],
            "ind_j": [],
            "mu_i": [],
            "mu_j": [],
            "offsets": [],
            "positions": [],
            "cell": [],
            "cell_map": [],
            "pair_map": [],
            "e_map": [],
        }

        cell = atoms.get_cell()
        dcell = np.linalg.pinv(cell)
        pos = atoms.get_scaled_positions()
        pos = np.matmul(pos, cell)
        n_pairs = 0
        for i in range(len(atoms)):
            at_nums = atoms.get_atomic_numbers()
            ind, off, dv = self.nl.get_neighbors(i)
            sort = np.argsort(ind)
            ind = ind[sort]
            dv = dv[sort]
            off = np.rint(
                np.dot(((dv + np.take(pos, [i] * len(ind), axis=0)) - pos[ind]), dcell)
            ).astype(np.int32)
            n_pairs += len(ind)
            self.data["ind_i"].append([i] * len(ind))
            self.data["ind_j"].append(ind)
            self.data["mu_i"].append([at_nums[i]] * len(ind))
            self.data["mu_j"].append(np.take(at_nums, ind))
            self.data["offsets"].append(off)
            self.data["e_map"].append([0])
            self.data["cell_map"].append([0])
        self.data["positions"].append(atoms.get_scaled_positions())
        self.data["cell"].append(cell.reshape(1, 3, 3))
        self.data["pair_map"].append([0] * n_pairs)

        self.data = {k: np.concatenate(v, axis=0) for k, v in self.data.items()}

    def calculate(
        self,
        atoms=None,
        properties=["energy", "forces", "stress"],
        system_changes=all_changes,
    ):
        Calculator.calculate(self, atoms, properties, system_changes)

        self.forces = np.empty((len(atoms), 3))
        self.stress = np.empty((1, 3, 3))
        self.energy = 0.0

        self.get_data(atoms)
        e, f, s = self.model._evaluate(*input2eval(self.data))

        self.energy, self.forces, self.stress = e.numpy(), f.numpy(), s.numpy()

        self.stress = np.matmul(
            self.stress.reshape(3, 3).astype(np.float64).transpose(1, 0),
            atoms.get_cell().astype(np.float64),
        ) / atoms.get_volume().astype(np.float64)
        self.results = {
            "energy": np.float64(
                self.energy.reshape(
                    -1,
                )
            ),
            "forces": self.forces.astype(np.float64),
            "stress": self.stress,
        }
