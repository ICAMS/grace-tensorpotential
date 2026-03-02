import numpy as np

from tensorpotential import constants


def _process_metric(
    key, value, metrics_dict, num_struct, num_atoms, normalization_spec
):
    """
    Process a single metric key-value pair and update the metrics dictionary.

    Args:
        key (str): The metric key from the accumulated metrics.
        value (float): The accumulated value of the metric.
        metrics_dict (dict): The dictionary to update with the processed metric.
        # mtype (str): The type of metric ('mae' or 'rmse').
        num_struct (int): The total number of structures.
        num_atoms (int): The total number of atoms.
        normalization_spec (dict, optional): Dictionary containing normalization rules.
    """

    if "abs" in key:
        mtype = "mae"
    elif "sqr" in key:
        mtype = "rmse"
    else:
        raise ValueError(f"Unknown metric type for {key=}")

    spec = normalization_spec[key]
    norm_basis = spec.get("norm")
    factor = spec.get("factor", 1.0)

    if norm_basis == "n_structures":
        divisor = num_struct * factor
    elif norm_basis == "n_atoms":
        divisor = num_atoms * factor
    else:
        raise ValueError(
            f"Unknown normalization type '{norm_basis=}' in normalization_spec for {key=}"
        )

    # Generate name based on key and mtype
    # e.g. abs/depa/per_struct -> mae/depa
    clean_key = key.replace("/per_struct", "")
    name = clean_key.replace("abs", mtype).replace("sqr", mtype)

    val = 0.0
    if divisor and divisor > 0:
        val = value / divisor

    if mtype == "rmse" and val >= 0:
        val = np.sqrt(val)
    metrics_dict[name] = val


def normalize_group_metrics(
    total_metrics, num_struct, num_atoms, n_batches, normalization_spec
):
    """
    Normalize accumulated metrics based on their type and count.

    Args:
        total_metrics (dict): Dictionary containing accumulated metrics.
        num_struct (int): Total number of structures.
        num_atoms (int): Total number of atoms.
        n_batches (int): Number of batches processed.
        normalization_spec (dict, optional): Dictionary containing normalization rules.

    Returns:
        dict: A dictionary of normalized metrics.
    """
    final_metrics = {}
    for k, v in total_metrics.items():
        if k in normalization_spec:
            _process_metric(
                k, v, final_metrics, num_struct, num_atoms, normalization_spec
            )
        elif "loss" in k:
            final_metrics[k] = float(v) / n_batches if n_batches > 0 else 0
        elif "total_time" in k:
            final_metrics[k + "/per_atom"] = v / num_atoms if num_atoms > 0 else 0
    return final_metrics


def process_accumulated_metrics(total_metrics, n_batches, normalization_spec=None):
    """
    Convert accumulated over batches metrics to total metric (by normalizing to proper number of observations).

    Args:
        total_metrics (dict): Dictionary containing accumulated metrics.
        n_batches (int): Number of batches processed.
        normalization_spec (dict, optional): Dictionary containing normalization rules.

    Returns:
        dict: A dictionary of final processed metrics.
    """
    num_struct = total_metrics.get("num_struct", 0)
    num_atoms = total_metrics.get("num_atoms", 0)

    final_metrics = normalize_group_metrics(
        total_metrics, num_struct, num_atoms, n_batches, normalization_spec
    )
    return final_metrics


def aggregate_metrics(step_results):
    """
    Aggregate metrics (with suffix /per_struct) for a single batch by summation.

    Args:
        step_results (dict): Dictionary of results of train/test step (contain metrics and other values)

    Returns:
        dict: Aggregated metrics dictionary containing 'num_atoms', 'num_struct', 'num_step',
              and summed values for per-structure metrics.
    """
    # TODO: implement structure_id aggregation, implement generation of per-structure-group metrics
    assert (
        "nat/per_struct" in step_results
    ), "Can't aggregate metrics without `nat/per_struct`"
    nat_per_struct = np.array(step_results["nat/per_struct"])
    out_dict = {}
    for k, v in step_results.items():
        if k == "nat/per_struct":
            out_dict["num_atoms"] = np.sum(v)
        elif k.endswith("/per_struct"):
            out_dict[k] = np.sum(v)
        elif "loss" in k or "jac" in k:
            if np.any(np.isnan(v)):
                print("NAN found!")
            out_dict[k] = np.array(v)  # per batch (already reduced over mini batches)
        else:
            pass  # raise ValueError(f"Unknown key to proceed: {k}")
    out_dict["num_struct"] = len(nat_per_struct)
    out_dict["num_step"] = 1
    return out_dict


def addup_metrics(metrics, acc_metrics):
    """
    Accumulate metrics over multiple batches.

    Args:
        metrics (dict): Metrics from the current batch.
        acc_metrics (dict): Dictionary to store accumulated metrics.

    Returns:
        dict: Updated accumulated metrics dictionary.
    """
    for k, v in metrics.items():
        if k in acc_metrics:
            acc_metrics[k] += v
        else:
            acc_metrics[k] = v
    return acc_metrics


def concatenate_per_structure_metrics(out, b_data, agg_concat_per_structure_metrics):
    if isinstance(b_data, list):  # manual dataset
        for b in b_data:
            # cur_structure_ids += list(b[constants.DATA_STRUCTURE_ID])
            agg_concat_per_structure_metrics[constants.DATA_STRUCTURE_ID] += [
                ind
                for ind in b[constants.DATA_STRUCTURE_ID].numpy().reshape(-1)
                if ind != -1
            ]
    elif isinstance(b_data, dict):  # tf.data.Dataset (distributed or not)
        # convert PerReplica to tensors and loop over
        try:
            it = b_data[constants.DATA_STRUCTURE_ID].values  # distributed
        except AttributeError:
            it = [b_data[constants.DATA_STRUCTURE_ID]]  # serial

        for structure_ind_tensor in it:
            agg_concat_per_structure_metrics[constants.DATA_STRUCTURE_ID] += [
                ind for ind in structure_ind_tensor.numpy().reshape(-1) if ind != -1
            ]

    # cur_metrics_per_struct_dict = defaultdict(list)
    for metric_name, metric_values in out.items():
        if metric_name.endswith("per_struct"):
            agg_concat_per_structure_metrics[metric_name] += list(
                metric_values.numpy().reshape(-1)
            )
