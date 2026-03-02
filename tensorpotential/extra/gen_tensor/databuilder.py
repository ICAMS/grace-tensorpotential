import numpy as np

from tensorpotential import constants
from tensorpotential.utils import enforce_pbc
from tensorpotential.extra.gen_tensor import constants as cc
from tensorpotential.data.databuilder import AbstractDataBuilder, get_padding_dims


class ReferenceTensorDataBuilder(AbstractDataBuilder):
    def __init__(
        self,
        normalize_weights=True,
        tensor_rank: int = 2,
        per_structure: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.normalize_weights = normalize_weights
        assert tensor_rank in [1, 2], "Only support tensor rank 1 or 2"
        self.tensor_rank = tensor_rank
        self.shape = 9 if self.tensor_rank == 2 else 3
        self.per_structure = per_structure

    def extract_from_ase_atoms(self, ase_atoms, **kwarg):
        raise NotImplementedError

    def extract_from_row(self, row, **kwarg):

        res = {
            cc.DATA_REFERENCE_TENSOR: np.array(row[cc.COLUMN_REFERENCE_TENSOR]).reshape(
                -1, self.shape
            ),
        }

        if constants.DATA_STRUCTURE_ID in kwarg:
            res[constants.DATA_STRUCTURE_ID] = kwarg[constants.DATA_STRUCTURE_ID]

        at = row[constants.COLUMN_ASE_ATOMS]

        if cc.COLUMN_REFERENCE_TENSOR_WEIGHT in row.index:
            res[cc.DATA_REFERENCE_TENSOR_WEIGHT] = np.array(
                row[cc.COLUMN_REFERENCE_TENSOR_WEIGHT]
            ).reshape(-1, 1)
        else:
            if self.per_structure:
                res[cc.DATA_REFERENCE_TENSOR_WEIGHT] = np.ones((1, 1))
            else:
                res[cc.DATA_REFERENCE_TENSOR_WEIGHT] = np.ones(len(at)).reshape(
                    -1, 1
                )  # per-atom

        return res

    def join_to_batch(self, pre_batch_list: list):
        res_dict = {}

        for key in [cc.DATA_REFERENCE_TENSOR]:
            data_list = [data_dict[key] for data_dict in pre_batch_list]
            res_dict[key] = (
                np.vstack(data_list).reshape(-1, self.shape).astype(self.float_dtype)
            )

        for key in [cc.DATA_REFERENCE_TENSOR_WEIGHT]:
            data_list = [data_dict[key] for data_dict in pre_batch_list]
            res_dict[key] = np.vstack(data_list).reshape(-1, 1).astype(self.float_dtype)

        for key in [constants.DATA_STRUCTURE_ID]:
            data_list = [int(data_dict[key]) for data_dict in pre_batch_list]
            res_dict[key] = np.array(data_list).reshape(-1, 1).astype(int)

        return res_dict

    def pad_batch(self, batch, max_pad_dict):
        pad_nat, pad_nneigh, pad_nstruct = get_padding_dims(batch, max_pad_dict)

        if pad_nat > 0:
            if not self.per_structure:
                k = cc.DATA_REFERENCE_TENSOR
                batch[k] = np.pad(
                    batch[k],
                    ((0, pad_nat), (0, 0)),
                    mode="constant",
                    constant_values=0,
                )

                k = cc.DATA_REFERENCE_TENSOR_WEIGHT
                batch[k] = np.pad(
                    batch[k], ((0, pad_nat), (0, 0)), mode="constant", constant_values=0
                )

        if pad_nstruct > 0:
            k = constants.DATA_STRUCTURE_ID
            batch[k] = np.pad(
                batch[k],
                ((0, pad_nstruct), (0, 0)),
                mode="constant",
                constant_values=-1,
            )
            if self.per_structure:
                k = cc.DATA_REFERENCE_TENSOR
                batch[k] = np.pad(
                    batch[k],
                    ((0, pad_nstruct), (0, 0)),
                    mode="constant",
                    constant_values=0,
                )

                k = cc.DATA_REFERENCE_TENSOR_WEIGHT
                batch[k] = np.pad(
                    batch[k],
                    ((0, pad_nstruct), (0, 0)),
                    mode="constant",
                    constant_values=0,
                )

    def postprocess_dataset(self, batches):
        if self.normalize_weights:
            weight_sum = np.sum(
                np.sum(b[cc.DATA_REFERENCE_TENSOR_WEIGHT]) for b in batches
            )

            for b in batches:
                b[cc.DATA_REFERENCE_TENSOR_WEIGHT] /= weight_sum


class PositionsDataBuilder(AbstractDataBuilder):
    def __init__(
        self,
        cutoff: float = 10.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.cutoff = cutoff

    def extract_from_ase_atoms(self, ase_atoms, **kwarg):
        ase_atoms = enforce_pbc(ase_atoms, cutoff=self.cutoff)
        return {constants.ATOMIC_POS: ase_atoms.get_positions()}

    def extract_from_row(self, row, **kwarg):
        return self.extract_from_ase_atoms(row["ase_atoms"])

    def join_to_batch(self, pre_batch_list: list):
        res_dict = {}

        data_list = [data_dict[constants.ATOMIC_POS] for data_dict in pre_batch_list]
        res_dict[constants.ATOMIC_POS] = (
            np.vstack(data_list).reshape(-1, 3).astype(self.float_dtype)
        )

        return res_dict

    def pad_batch(self, batch, max_pad_dict):
        pad_nat, pad_nneigh, pad_nstruct = get_padding_dims(batch, max_pad_dict)

        if pad_nat > 0:
            k = constants.ATOMIC_POS
            batch[k] = np.pad(
                batch[k],
                ((0, pad_nat), (0, 0)),
                mode="constant",
                constant_values=0,
            )


class CellDataBuilder(AbstractDataBuilder):
    def __init__(
        self,
        cutoff: float = 10.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.cutoff = cutoff

    def extract_from_ase_atoms(self, ase_atoms, **kwarg):
        ase_atoms = enforce_pbc(ase_atoms, cutoff=self.cutoff)
        return {constants.CELL_VECTORS: ase_atoms.get_cell()}

    def extract_from_row(self, row, **kwarg):
        return self.extract_from_ase_atoms(row["ase_atoms"])

    def join_to_batch(self, pre_batch_list: list):
        res_dict = {}

        data_list = [data_dict[constants.CELL_VECTORS] for data_dict in pre_batch_list]
        res_dict[constants.CELL_VECTORS] = (
            np.vstack(data_list).reshape(-1, 3, 3).astype(self.float_dtype)
        )

        return res_dict

    def pad_batch(self, batch, max_pad_dict):
        pad_nat, pad_nneigh, pad_nstruct = get_padding_dims(batch, max_pad_dict)

        if pad_nstruct > 0:
            k = constants.CELL_VECTORS
            batch[k] = np.pad(
                batch[k],
                ((0, pad_nstruct), (0, 0), (0, 0)),
                mode="constant",
                constant_values=0.1,
            )
