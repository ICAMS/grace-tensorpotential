from __future__ import annotations

from typing import Final


GPU_INDEX: Final[str] = "gpu_ind"
GPU_MEMORY_LIMIT: Final[str] = "mem_limit"


# Structural
N_NEIGHBORS_REAL: Final[str] = "n_neigh_real"

N_ATOMS_BATCH_REAL: Final[str] = "batch_tot_nat_real"
N_ATOMS_BATCH_TOTAL: Final[str] = "batch_tot_nat"

N_STRUCTURES_BATCH_REAL: Final[str] = "batch_total_num_structures"
N_STRUCTURES_BATCH_TOTAL: Final[str] = "n_struct_total"

ATOMIC_MU_I: Final[str] = "atomic_mu_i"
ATOMIC_MAGMOM: Final[str] = "atomic_magmom"
ATOMS_TO_STRUCTURE_MAP: Final[str] = "map_atoms_to_structure"
BONDS_TO_STRUCTURE_MAP: Final[str] = "map_bonds_to_structure"

BOND_VECTOR: Final[str] = "bond_vector"
BOND_MU_I: Final[str] = "mu_i"
BOND_MU_J: Final[str] = "mu_j"
BOND_IND_I: Final[str] = "ind_i"
BOND_IND_J: Final[str] = "ind_j"

PAD_MAX_N_STRUCTURES: Final[str] = "max_nstruct"
PAD_MAX_N_ATOMS: Final[str] = "max_nat"
# TODO: Possibly rename things related to n_neighbors -> n_bonds?
PAD_MAX_N_NEIGHBORS: Final[str] = "max_nneigh"


# Properties
DATA_REFERENCE_ENERGY: Final[str] = "true_energy"
DATA_REFERENCE_FORCES: Final[str] = "true_force"
DATA_REFERENCE_VIRIAL: Final[str] = "true_virial"

DATA_ENERGY_WEIGHTS: Final[str] = "energy_weight"
DATA_FORCE_WEIGHTS: Final[str] = "force_weight"
DATA_VIRIAL_WEIGHTS: Final[str] = "virial_weight"
DATA_VOLUME: Final[str] = "volume"

DATA_STRUCTURE_ID: Final[str] = "structure_id"

# Predict
PREDICT_ATOMIC_ENERGY: Final[str] = "atomic_energy"
PREDICT_TOTAL_ENERGY: Final[str] = "total_energy"
PREDICT_FORCES: Final[str] = "total_f"
PREDICT_VIRIAL: Final[str] = "virial"
PREDICT_PAIR_FORCES: Final[str] = "pair_f"

L2_LOSS_COMPONENT: Final[str] = "l2_loss_component"

########### INPUT.YAML CONSTANTS #############
INPUT_CUTOFF: Final[str] = "cutoff"
INPUT_CUTOFF_DICT: Final[str] = "cutoff_dict"

INPUT_DATA_SECTION: Final[str] = "data"
INPUT_POTENTIAL_SECTION: Final[str] = "potential"
INPUT_FIT_SECTION: Final[str] = "fit"

INPUT_FIT_LOSS: Final[str] = "loss"
INPUT_FIT_LOSS_ENERGY: Final[str] = "energy"
INPUT_FIT_LOSS_FORCES: Final[str] = "forces"
INPUT_FIT_LOSS_STRESS: Final[str] = "stress"
INPUT_FIT_LOSS_VIRIAL: Final[str] = "virial"
INPUT_USE_PER_SPECIE_N_NEI: Final[str] = "use_per_specie_n_nei"


INPUT_REFERENCE_ENERGY: Final[str] = "reference_energy"

COLUMN_ATOMIC_MAGMOM: Final[str] = "mag_mom"
COLUMN_ASE_ATOMS: Final[str] = "ase_atoms"

PLACEHOLDER: Final[str] = "placeholder"
