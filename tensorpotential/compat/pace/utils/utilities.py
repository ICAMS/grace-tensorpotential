import numpy as np
import tensorflow as tf

from ase.neighborlist import NewPrimitiveNeighborList
from tensorpotential.compat.pace.constants import *


def split_dataset(data_set, shuffle=True, n_train=1000, n_test=500):
    if (n_train + n_test) > len(data_set):
        print("Data set does not have enough entries!")

    index = np.arange(0, len(data_set))
    if shuffle:
        np.random.seed(100)
        np.random.shuffle(index)

    index_train = index[:n_train]
    index_test = index[n_train : n_train + n_test]

    return [data_set[i] for i in index_train], [data_set[i] for i in index_test]


def apply_weights(row, fname="w_forces", ename="w_energy"):
    row["tp_atoms"]["_eweights"] = np.array([row[ename]]).reshape(-1, 1)
    row["tp_atoms"]["_fweights"] = np.array(row[fname]).reshape(-1, 1)


def batching_data(df, batch_size=10):
    df.dropna(subset=["tp_atoms"], inplace=True)
    try:
        df.apply(apply_weights, axis=1)
    except:
        pass
    df.index = np.arange(len(df))
    batch_index = chunks(len(df), batch_size)
    batches = []
    for batch in batch_index:
        batch_list = list(df.loc[batch]["tp_atoms"])
        batches.append(supply_batch(batch_list))

    return batches


def check_or_put_unit_weights(dic):
    if "_eweights" not in list(dic):
        dic.update({"_eweights": np.ones([len(dic["_energy"]), 1])})

    if "_fweights" not in list(dic):
        dic.update({"_fweights": np.ones([len(dic["_forces"]), 1])})

    return dic


def chunks(size, chunksize):
    n = max(1, chunksize)
    lst = np.arange(0, size)
    return [lst[i : i + n] for i in range(0, size, n)]


def supply_batch(data):
    batch_data = {
        "ind_i": [],
        "ind_j": [],
        "mu_i": [],
        "mu_j": [],
        "offsets": [],
        "energy": [],
        "forces": [],
        "positions": [],
        "eweights": [],
        "fweights": [],
        "cell": [],
        "cell_map": [],
        "pair_map": [],
        "e_map": [],
    }

    count_atoms = 0
    count_cells = 0
    for j, entry in enumerate(data):
        entry = check_or_put_unit_weights(entry)
        n_atoms = entry["_positions"].shape[0]
        n_pairs = entry["_ind_j"].shape[0]

        batch_data["ind_i"].append(entry["_ind_i"] + count_atoms)
        batch_data["ind_j"].append(entry["_ind_j"] + count_atoms)
        batch_data["mu_i"].append(entry["_mu_i"])
        batch_data["mu_j"].append(entry["_mu_j"])
        batch_data["offsets"].append(entry["_offsets"])
        batch_data["energy"].append(entry["_energy"])
        batch_data["forces"].append(entry["_forces"])
        batch_data["eweights"].append(entry["_eweights"])
        batch_data["fweights"].append(entry["_fweights"])
        batch_data["positions"].append(entry["_positions"])
        batch_data["cell"].append(entry["_cell"])
        batch_data["pair_map"].append([count_cells] * n_pairs)
        batch_data["cell_map"].append([count_cells] * n_atoms)
        batch_data["e_map"].append([j] * n_atoms)

        count_atoms += n_atoms
        count_cells += 1

    batch_data = {k: np.concatenate(v, axis=0) for k, v in batch_data.items()}

    return batch_data


def enforce_pbc(atoms, cutoff):
    pos = atoms.get_positions()
    if (atoms.get_pbc() == 0).all():
        max_d = np.max(np.linalg.norm(pos - pos[0], axis=1))
        cell = np.eye(3) * ((max_d + cutoff) * 2)
        atoms.set_cell(cell)
        atoms.center()

    return atoms


def get_nghbrs(atoms, list_at_ind=None, cutoff=8.7, skin=0, verbose=False):
    nghbrs_lst = NewPrimitiveNeighborList(
        cutoffs=[cutoff * 0.5] * len(atoms),
        skin=skin,
        self_interaction=False,
        bothways=True,
        use_scaled_positions=True,
    )

    nghbrs_lst.update(atoms.get_pbc(), atoms.get_cell(), atoms.get_scaled_positions())
    # cell = atoms.get_cell()
    ind_i = []
    mu_i = []
    ind_j = []
    mu_j = []
    offsts = []
    at_nums = atoms.get_atomic_numbers()
    if list_at_ind is not None:
        list_of_atoms = list_at_ind
    else:
        list_of_atoms = np.arange(0, len(atoms))
    check = 0
    for i in list_of_atoms:
        ind, off = nghbrs_lst.get_neighbors(i)
        if len(ind) < 1:
            check += 1

        sort = np.argsort(ind)
        ind = ind[sort]
        off = off[sort]
        ind_i.append([i] * len(ind))
        mu_i.append([at_nums[i]] * len(ind))
        ind_j.append(ind)
        mu_j.append(np.take(at_nums, ind))
        offsts.append(off)
    if check == 0:
        return (
            np.hstack(ind_i),
            np.hstack(ind_j),
            np.hstack(mu_i),
            np.hstack(mu_j),
            np.vstack(offsts).astype(np.float64),
        )
    else:
        if verbose:
            print(
                "Found an atom with no neighbors within cutoff. This structure will be skipped"
            )
        return None


def copy_atoms(atoms):
    if atoms.get_calculator() is not None:
        calc = atoms.get_calculator()
        new_atoms = atoms.copy()
        new_atoms.set_calculator(calc)
    else:
        new_atoms = atoms.copy()

    return new_atoms


def generate_tp_atoms(ase_atoms, list_at_ind=None, cutoff=8.7, verbose=True):
    atoms = copy_atoms(ase_atoms)
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    atoms = enforce_pbc(atoms, cutoff)
    env = get_nghbrs(atoms, list_at_ind=list_at_ind, cutoff=cutoff, verbose=verbose)
    positions = atoms.get_scaled_positions()
    if env is not None:
        cell = atoms.get_cell().reshape(1, 3, 3)

        data = {}
        data["_ind_i"] = env[0]
        data["_ind_j"] = env[1]
        data["_mu_i"] = env[2]
        data["_mu_j"] = env[3]
        data["_offsets"] = env[4]
        data["_eweights"] = np.ones((1, 1))
        data["_fweights"] = np.ones((len(atoms), 1))
        data["_energy"] = np.array(energy).reshape(-1, 1)
        data["_forces"] = forces.astype(np.float64)
        data["_positions"] = positions.astype(np.float64)
        data["_cell"] = cell.astype(np.float64)

        return data
    else:
        return None


def input2eval(df):
    return (
        tf.convert_to_tensor(df["positions"], dtype=tf.float64, name="positions"),
        tf.convert_to_tensor(df["offsets"], dtype=tf.float64, name="offsets"),
        tf.convert_to_tensor(df["cell"], dtype=tf.float64, name="cell"),
        tf.convert_to_tensor(df["ind_i"], dtype=tf.int32, name="ind_i"),
        tf.convert_to_tensor(df["ind_j"], dtype=tf.int32, name="ind_j"),
        tf.convert_to_tensor(df["mu_i"], dtype=tf.int32, name="mu_i"),
        tf.convert_to_tensor(df["mu_j"], dtype=tf.int32, name="mu_j"),
        tf.convert_to_tensor(df["pair_map"], dtype=tf.int32, name="pair_map"),
        tf.convert_to_tensor(df["cell_map"], dtype=tf.int32, name="cell_map"),
        tf.convert_to_tensor(df["e_map"], dtype=tf.int32, name="e_map"),
    )


def input2evaloss(df):
    return (
        tf.convert_to_tensor(df["positions"], dtype=tf.float64, name="positions"),
        tf.convert_to_tensor(df["offsets"], dtype=tf.float64, name="offsets"),
        tf.convert_to_tensor(df["cell"], dtype=tf.float64, name="cell"),
        tf.convert_to_tensor(df["energy"], dtype=tf.float64, name="energy"),
        tf.convert_to_tensor(df["forces"], dtype=tf.float64, name="forces"),
        tf.convert_to_tensor(df["eweights"], dtype=tf.float64, name="eweights"),
        tf.convert_to_tensor(df["fweights"], dtype=tf.float64, name="fweights"),
        tf.convert_to_tensor(df["ind_i"], dtype=tf.int32, name="ind_i"),
        tf.convert_to_tensor(df["ind_j"], dtype=tf.int32, name="ind_j"),
        tf.convert_to_tensor(df["mu_i"], dtype=tf.int32, name="mu_i"),
        tf.convert_to_tensor(df["mu_j"], dtype=tf.int32, name="mu_j"),
        tf.convert_to_tensor(df["pair_map"], dtype=tf.int32, name="pair_map"),
        tf.convert_to_tensor(df["cell_map"], dtype=tf.int32, name="cell_map"),
        tf.convert_to_tensor(df["e_map"], dtype=tf.int32, name="e_map"),
    )


def input2evalbasis(df):
    try:
        ref_basis = tf.convert_to_tensor(df["ref_basis"], dtype=tf.float64)
    except:
        ref_basis = tf.reshape(
            tf.convert_to_tensor(np.zeros((1, 1)), dtype=tf.float64), [1, 1]
        )
    return (
        tf.convert_to_tensor(df["positions"], dtype=tf.float64),
        tf.convert_to_tensor(df["offsets"], dtype=tf.float64),
        tf.convert_to_tensor(df["cell"], dtype=tf.float64),
        tf.convert_to_tensor(df["energy"], dtype=tf.float64),
        tf.convert_to_tensor(df["forces"], dtype=tf.float64),
        tf.convert_to_tensor(df["eweights"], dtype=tf.float64),
        tf.convert_to_tensor(df["fweights"], dtype=tf.float64),
        tf.convert_to_tensor(df["ind_i"], dtype=tf.int32),
        tf.convert_to_tensor(df["ind_j"], dtype=tf.int32),
        tf.convert_to_tensor(df["mu_i"], dtype=tf.int32),
        tf.convert_to_tensor(df["mu_j"], dtype=tf.int32),
        tf.convert_to_tensor(df["pair_map"], dtype=tf.int32),
        tf.convert_to_tensor(df["cell_map"], dtype=tf.int32),
        tf.convert_to_tensor(df["e_map"], dtype=tf.int32),
        ref_basis,
    )


def _set_gpu_config(config=None):
    conf_dict = {}
    conf_dict[GPU_INDEX] = 0
    conf_dict[GPU_MEMORY_LIMIT] = 0

    if config is not None:
        assert isinstance(config, dict), "gpu_config must be a dict"
        if GPU_INDEX in config:
            assert isinstance(config[GPU_INDEX], int), "{} must be an integer".format(
                GPU_INDEX
            )
            conf_dict[GPU_INDEX] = config[GPU_INDEX]

        if GPU_MEMORY_LIMIT in config:
            assert isinstance(
                config[GPU_MEMORY_LIMIT], int
            ), "{} must be an integer number of MB".format(GPU_MEMORY_LIMIT)
            if config[GPU_MEMORY_LIMIT] < 0:
                raise ValueError("{} must be not negative".format(GPU_MEMORY_LIMIT))
            if conf_dict[GPU_INDEX] >= 0 and config[GPU_MEMORY_LIMIT] > 0:
                try:
                    total_gpu_mem = get_gpu_memory(conf_dict[GPU_INDEX])
                except:
                    total_gpu_mem = 0
                assert (
                    config[GPU_MEMORY_LIMIT] <= total_gpu_mem
                ), "Requested GPU memory limit is greater than total GPU memory"
                conf_dict[GPU_MEMORY_LIMIT] = config[GPU_MEMORY_LIMIT]

    return conf_dict


def init_gpu_config(gpu_config):
    gpu_config = _set_gpu_config(gpu_config)
    if gpu_config[GPU_INDEX] < 0:
        sel_gpus = []
        try:
            tf.config.set_visible_devices(sel_gpus, "GPU")
            tf.config.list_logical_devices("GPU")
        except RuntimeError as e:
            print(e)
    elif gpu_config[GPU_INDEX] >= 0:
        avail_gpus = tf.config.list_physical_devices("GPU")
        if len(avail_gpus) > 0:
            assert gpu_config[GPU_INDEX] < len(
                avail_gpus
            ), "GPU ind {} is requested, but there are only {} GPUs".format(
                gpu_config[GPU_INDEX], len(avail_gpus)
            )
            sel_gpus = avail_gpus[gpu_config[GPU_INDEX]]
            if gpu_config[GPU_MEMORY_LIMIT] != 0:
                try:
                    tf.config.set_logical_device_configuration(
                        sel_gpus,
                        [
                            tf.config.LogicalDeviceConfiguration(
                                memory_limit=gpu_config[GPU_MEMORY_LIMIT]
                            )
                        ],
                    )
                    logical_gpus = tf.config.list_logical_devices("GPU")
                except RuntimeError as e:
                    print(e)
            else:
                try:
                    tf.config.set_visible_devices(sel_gpus, "GPU")
                    logical_gpus = tf.config.list_logical_devices("GPU")
                except RuntimeError as e:
                    print(e)

    return gpu_config


def get_gpu_memory(gpu_id):
    import subprocess as sp

    command = f"nvidia-smi --id={gpu_id} --query-gpu=memory.total --format=csv"
    output_cmd = sp.check_output(command.split())
    memory = output_cmd.decode("ascii").split("\n")[1]
    memory = int(memory.split()[0])

    return memory
