from __future__ import annotations

import inspect
import logging
import warnings
from abc import ABC, abstractmethod
from enum import Enum
from typing import Iterable, Literal, Set, Callable, List, Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
import tensorflow as tf
import yaml
from tensorflow.python.trackable.data_structures import ListWrapper

# arg-type for TPInstruction.summary()
summary_verbose_type = Literal[0, 1, 2]


class InstructionManager:
    def __init__(self):
        self.instruction_dict: dict = {}

    def __enter__(self):
        global _instruction_manager
        _instruction_manager = self
        return self

    def __exit__(self, *args, **kwargs):
        global _instruction_manager
        _instruction_manager = None

    def get_instructions(self) -> dict[str, TPInstruction]:
        return self.instruction_dict


class NoInstructionManager:
    def __init__(self):

        self.orig_ins_manager = None

    def __enter__(self):
        global _instruction_manager
        self.orig_ins_manager = _instruction_manager
        _instruction_manager = None
        return self

    def __exit__(self, *args, **kwargs):
        global _instruction_manager
        _instruction_manager = self.orig_ins_manager


_instruction_manager: Optional[InstructionManager] = None


def class_to_str(cls):
    return cls.__module__ + "." + cls.__qualname__


def str_to_class(class_name):
    if class_name is None:
        return None
    module = ".".join(class_name.split(".")[:-1])
    name = class_name.split(".")[-1]
    context = {}
    if module:
        exec(
            "from {module} import {name} as EXTRACTED_C; EXTRACTED_C".format(
                module=module, name=name
            ),
            context,  # Pass the context dictionary explicitly
        )
    else:
        raise ValueError("Couldn't deserialize class `{}`".format(class_name))
    if "EXTRACTED_C" not in context:
        raise ValueError("Couldn't deserialize class `{}`".format(class_name))
    return context["EXTRACTED_C"]


def recursive_walk_and_modify(obj, callback):
    """
    Recursively walks over lists and dictionaries, modifying elements in-place based on a callback function.

    Args:
        obj: The list or dictionary to walk over.
        callback: callback(k | index, v, container) The function to modify each element (including nested).

    Returns:
        None
    """

    if isinstance(obj, list):
        for k, v in enumerate(obj):
            callback(k, v, obj)  # Pass the list for in-place modification
            v = obj[k]
            recursive_walk_and_modify(v, callback)
    elif isinstance(obj, dict):
        for k, v in obj.items():  # Create a copy to avoid modification errors
            callback(k, v, obj)
            v = obj[k]
            recursive_walk_and_modify(v, callback)
    else:
        pass  # No need to modify other data types


def replace_TPInstruction_to_name(k, v, container):
    if isinstance(v, TPInstruction):
        container[k] = {"_instruction_": True, "name": v.name}
    elif isinstance(v, dict):
        container[k] = dict(v)
    elif isinstance(v, (list, tuple)):
        container[k] = list(v)
    elif isinstance(v, np.floating):
        container[k] = float(v)
    elif isinstance(v, np.integer):
        container[k] = int(v)
    elif not isinstance(v, (int, str, bool, float)):
        if v is not None:
            warnings.warn(
                f"Unrecognized type {type(v)} for TPInstruction serialization"
            )


def capture_init_args(cls):
    """Decorator to capture arguments passed to a class's __init__ method, including defaults and
    add to_dict and from_dict methods

       Args:
           cls (class): The class to decorate.

       Returns:
           class: The decorated class with a new __init__ method that stores the arguments as a dict.
    """
    original_init = cls.__init__
    num_defaults = len(original_init.__defaults__) if original_init.__defaults__ else 0
    non_default_param_names = cls.__init__.__code__.co_varnames[
        1 : cls.__init__.__code__.co_argcount - num_defaults
    ]

    s = inspect.signature(original_init)
    default_values = {
        name: p.default
        for name, p in s.parameters.items()
        if name != "self" and p.default != inspect._empty
    }
    if original_init.__kwdefaults__ is not None:
        default_values.update(original_init.__kwdefaults__)

    def __init__(self, *args, **kwargs):
        """Wrapper __init__ method that captures arguments and includes defaults.

        Combines positional and keyword arguments with defaults into a single dictionary.

        Args:
            self (object): The instance of the class.
            *args: Positional arguments passed to the class.
            **kwargs: Keyword arguments passed to the class.
        """

        init_args = {}
        init_args.update(default_values)
        init_args.update(
            dict(zip(non_default_param_names, args[: len(non_default_param_names)]))
        )
        init_args.update(kwargs)
        if not hasattr(self, "_init_args"):
            self._init_args = {}
        self._init_args.update(
            init_args
        )  # to avoid rewriting collected init_args by child/parent classes
        original_init(self, *args, **kwargs)

    def to_dict(self):
        dct = {"__cls__": class_to_str(self.__class__)}
        dct.update(self._init_args)
        # Convert TPInstruction to its name:
        recursive_walk_and_modify(dct, replace_TPInstruction_to_name)
        # TODO: call pre_serialize ?
        return dct

    def from_dict(cls, dct):
        dct = dct.copy()
        if "__cls__" in dct:
            cls = str_to_class(dct.pop("__cls__"))
        # TODO: call pre_deserialize ?
        # print(f"Deserialize class {cls} with args {dct}")
        return cls(**dct)

    cls.__init__ = __init__
    cls.to_dict = to_dict
    cls.from_dict = classmethod(from_dict)
    return cls


def save_instructions_dict(
    filename: str,
    instructions_list_or_dict: list[TPInstruction] | dict[str, TPInstruction],
):
    """
    Enforce the conversion to dict of instructions and save into YAML
    """

    instructions_dict: dict[str, TPInstruction] = (
        {ins.name: ins for ins in instructions_list_or_dict}
        if isinstance(instructions_list_or_dict, list)
        else instructions_list_or_dict
    )

    dict_of_instructions_dict: dict[str, dict] = {}

    for name, ins in instructions_dict.items():
        dct = ins.to_dict()
        # process TF's dicts wrappers and TPInstructions, call second time, just for the case
        recursive_walk_and_modify(dct, replace_TPInstruction_to_name)
        dict_of_instructions_dict[name] = dct
    with open(filename, "w") as f:
        yaml.dump(dict_of_instructions_dict, stream=f, sort_keys=False)


def load_instructions(filename: str):
    """Load serialized instructions from YAML (either as list or dict of instructions)"""
    with open(filename, "rt") as f:
        collection_of_dict_instructions = yaml.safe_load(
            f,
        )

    deserialized_instructions_dict = {}

    # custom callback, that capture deserialized_instructions_dict
    def tp_instruction_replace(key, item, container):
        if isinstance(item, dict) and "_instruction_" in item:
            name = item["name"]
            container[key] = deserialized_instructions_dict[name]

    is_list = True  # by default - instructions list
    if isinstance(collection_of_dict_instructions, dict):
        collection_of_dict_instructions = list(collection_of_dict_instructions.values())
        is_list = False

    for ins_dict in collection_of_dict_instructions:
        try:
            recursive_walk_and_modify(ins_dict, tp_instruction_replace)
            ins = TPInstruction.from_dict(ins_dict)
            deserialized_instructions_dict[ins.name] = ins
        except TypeError as e:
            logging.error(f"Can't load instruction from {ins_dict=}")
            raise e

    # convert to list
    instructions = (
        list(deserialized_instructions_dict.values())
        if is_list
        else deserialized_instructions_dict
    )

    return instructions


@capture_init_args
class TPInstruction(tf.Module, ABC):
    _INSTRUCTION_TYPE = "TPInstruction"

    def __init__(self, name="TPInstruction"):
        super().__init__(name=name)
        self.is_built = False
        self.training = False
        self._register_instruction_in_context()
        # TODO: KOSTYL'!!!
        self.n_out = None

    def _register_instruction_in_context(self):
        global _instruction_manager
        if _instruction_manager is not None:
            # check for unique name
            if self.name in _instruction_manager.instruction_dict:
                raise RuntimeError(
                    f"Instruction name {self.name} already used by {_instruction_manager.instruction_dict[self.name]}"
                )
            _instruction_manager.instruction_dict[self.name] = self

    @abstractmethod
    def frwrd(self, input_data: dict, training: bool = False):
        pass

    @abstractmethod
    def build(self, float_dtype):
        pass

    def get_variables_info(self) -> dict:
        info = {}
        info["instruction"] = self.name
        for var in self.trainable_variables:
            info[var.name] = var.get_shape().as_list()
        return info

    # @abstractmethod
    # def from_dict(self, *args, **kwargs):
    #     pass
    #
    # @abstractmethod
    # def to_dict(self, *args, **kwargs):
    #     pass

    @classmethod
    def get_instruction_type(cls) -> str:
        return cls._INSTRUCTION_TYPE

    @tf.Module.with_name_scope
    def __call__(self, input_data: dict, training: bool = False) -> dict:
        output = self.frwrd(input_data, training=training)
        assert self.name not in input_data, f"{self.name} already exists in input"

        input_data[self.name] = output

        return input_data

    def summary(
        self, verbose: summary_verbose_type, shown_vars: Set[str] = None
    ) -> str:
        return InstructionPrinter.get_summary(self, verbose, shown_vars)

    def __repr__(self):
        return InstructionPrinter.repr_instruction(self, verbose=1)


class LORAInstructionMixin(ABC):

    def __init__(self, lora_config):
        self.lora = False
        self.lora_config = None

        if lora_config:
            self.lora = True
            self.lora_config = lora_config

    @abstractmethod
    def enable_lora_adaptation(self, lora_config: dict[str, Any]):
        self.lora_config = lora_config
        self.lora = True
        self._init_args["lora_config"] = lora_config

    @abstractmethod
    def finalize_lora_update(self):
        # common part
        self.lora = False
        self.lora_config = None
        if "lora_config" in self._init_args:
            del self._init_args["lora_config"]


class ElementsReduceInstructionMixin(ABC):

    @abstractmethod
    def prepare_variables_for_selected_elements(self, index_to_select):
        raise NotImplementedError()

    @abstractmethod
    def upd_init_args_new_elements(self, new_element_map):
        raise NotImplementedError()


class TPEquivariantInstruction(TPInstruction):
    _INSTRUCTION_TYPE = "TPEquivariantInstruction"

    def __init__(
        self,
        lmax,
        coupling_meta_data: pd.DataFrame = None,
        coupling_origin: list[str, str] = None,
        name="TPEquivariantInstruction",
    ):
        super().__init__(name=name)
        self.lmax = lmax
        self.coupling_meta_data = coupling_meta_data
        self.coupling_origin = coupling_origin

    @abstractmethod
    def build(self, float_dtype):
        pass

    @abstractmethod
    def frwrd(self, input_data, training=False):
        pass

    def init_uncoupled_meta_data(self, l_p_init_list: list[list] = None):
        if l_p_init_list is None:
            l_p_init_list = [[l, 1 if l % 2 == 0 else -1] for l in range(self.lmax + 1)]
        meta = []
        for l, p in l_p_init_list:
            for m in range(-l, l + 1):
                meta.append([l, m, "", p, l])
        meta_df = pd.DataFrame(meta, columns=["l", "m", "hist", "parity", "sum_of_ls"])

        return meta_df

    @staticmethod
    def collect_functions_from_meta_data(
        coupling_meta_data: pd.DataFrame, max_l: int, l_p_list: list[list] = None
    ) -> dict:
        if l_p_list is None:
            plist = []
            for l in range(max_l + 1):
                # p = 1 if l % 2 == 0 else -1
                plist.append([l, 1])
                plist.append([l, -1])
        else:
            plist = l_p_list

        ind_d = coupling_meta_data.groupby(["parity", "l", "hist"]).indices

        func_ids = [
            v for (p, l, *_), v in ind_d.items() if l <= max_l and [l, p] in plist
        ]

        collect_d = {}
        ind = np.concatenate(func_ids)
        collect_d["func_collect_ind"] = ind
        collect_meta_df = coupling_meta_data.iloc[ind]
        collect_meta_df.reset_index(inplace=True)
        collect_d["collect_meta_df"] = collect_meta_df

        collect_ind = collect_meta_df.groupby(["parity", "l", "hist"]).indices
        w_shape = len(collect_ind)
        collect_d["w_shape"] = w_shape
        w_l_tile = np.zeros(len(collect_meta_df))
        w_l_tile[np.concatenate([v for k, v in collect_ind.items()])] = np.concatenate(
            [[i] * len(v) for i, (k, v) in enumerate(collect_ind.items())]
        )
        collect_d["w_l_tile"] = tf.constant(w_l_tile, dtype=tf.int32)

        return collect_d

    def collect_functions(self, max_l: int, l_p_list: list[list] = None) -> dict:
        """
        Finds functions in the current tensor that have l <= max_l
        and have fitting [l, p] values

        :param max_l: maximum l of the function to collect
        :param l_p_list: admissible [l, p] combinations for a function
        :return: collection maps
        """

        if l_p_list is None:
            plist = []
            for l in range(max_l + 1):
                # p = 1 if l % 2 == 0 else -1
                plist.append([l, 1])
                plist.append([l, -1])
        else:
            plist = l_p_list

        ind_d = self.coupling_meta_data.groupby(["parity", "l", "hist"]).indices

        # TODO: this list is empty sometimes, for a given instruction and restricted [l, p].
        #  Need to treat it properly below, but more importantly in the full context of the collector.
        #  For now, make sure the [l, p] list is not too restrictive.
        func_ids = [
            v for (p, l, *_), v in ind_d.items() if l <= max_l and [l, p] in plist
        ]

        collect_d = {}
        ind = np.concatenate(func_ids)
        collect_d["func_collect_ind"] = ind
        collect_meta_df = self.coupling_meta_data.iloc[ind]
        collect_meta_df.reset_index(inplace=True)
        collect_d["collect_meta_df"] = collect_meta_df

        collect_ind = collect_meta_df.groupby(["parity", "l", "hist"]).indices
        w_shape = len(collect_ind)
        collect_d["w_shape"] = w_shape
        w_l_tile = np.zeros(len(collect_meta_df))
        w_l_tile[np.concatenate([v for k, v in collect_ind.items()])] = np.concatenate(
            [[i] * len(v) for i, (k, v) in enumerate(collect_ind.items())]
        )
        collect_d["w_l_tile"] = tf.constant(w_l_tile, dtype=tf.int32)

        return collect_d

    def select_functions(self, selected_l: int, selected_p: -1 | 1) -> dict:
        """
        Finds functions in the current tensor that have l == selected_l
        and p == selected_p

        :param selected_l: value of l of the function to select
        :param selected_p: admissible parity of the function
        :return: selection maps
        """

        ind_d = self.coupling_meta_data.groupby(["parity", "l", "hist"]).indices

        func_ids = [
            v for (p, l, *_), v in ind_d.items() if l == selected_l and p == selected_p
        ]

        collect_d = {}
        ind = np.concatenate(func_ids)
        collect_d["func_collect_ind"] = ind
        collect_meta_df = self.coupling_meta_data.iloc[ind]
        collect_meta_df.reset_index(inplace=True)
        collect_d["collect_meta_df"] = collect_meta_df

        collect_ind = collect_meta_df.groupby(["parity", "l", "hist"]).indices
        w_shape = len(collect_ind)
        collect_d["w_shape"] = w_shape
        w_l_tile = np.zeros(len(collect_meta_df))
        w_l_tile[np.concatenate([v for k, v in collect_ind.items()])] = np.concatenate(
            [[i] * len(v) for i, (k, v) in enumerate(collect_ind.items())]
        )
        collect_d["w_l_tile"] = tf.constant(w_l_tile, dtype=tf.int32)

        return collect_d


class InstructionPrinter:
    @staticmethod
    def get_summary(
        instruction: TPInstruction,
        verbose: summary_verbose_type = 0,
        shown_vars: Set[str] = None,
    ) -> str:
        res_str = InstructionPrinter.repr_instruction(instruction, verbose)
        res_str += InstructionPrinter.repr_var_info(instruction, shown_vars)

        if verbose == 1 and hasattr(instruction, "coupling_meta_data"):
            res_str += InstructionPrinter.repr_coupling_df(
                InstructionPrinter._groupby_and_count_Aterms(
                    instruction.coupling_meta_data
                ),
                "Coupling Functions",
            )

        if verbose == 2 and hasattr(instruction, "coupling_meta_data"):
            df = instruction.coupling_meta_data
            if "cg_list" in df.columns:
                df = InstructionPrinter._format_coupling_cg_list(df)
            res_str += InstructionPrinter.repr_coupling_df(df, "Coupling Table")

        return res_str

    @staticmethod
    def is_instruction_list(obj: ListWrapper) -> bool:
        return (
            isinstance(obj, ListWrapper)
            and len(obj) > 0
            and isinstance(obj[0], TPInstruction)
        )

    @staticmethod
    def is_list_of_lists(obj: ListWrapper) -> bool:
        # pycharm says lw-type should be Iterable instead of ListWrapper, but it's mistaken. Probably because it's tf type
        return (
            isinstance(obj, ListWrapper) and len(obj) > 0 and isinstance(obj[0], list)
        )

    @staticmethod
    def repr_instruction(instruction: TPInstruction, verbose: int):
        res_str = f"{instruction.name} = {instruction.__class__.__name__}"
        if verbose == 0:
            return res_str

        _init_args = instruction._init_args
        res_str += f"(name={_init_args['name']}, "
        for k, v in _init_args.items():
            if k == "name" or v is None:
                continue
            if k == "keep_parity":
                res_str += f"{k}={ParityType.get_parity_type(v).name}, "
                continue
            if isinstance(v, TPInstruction):
                res_str += f"{k}={v.name}, "
                continue
            if isinstance(v, ListWrapper):
                res_str += f"{k}={InstructionPrinter.repr_list_wrapper(v)}, "
                continue
            res_str += f"{k}={v}, "
        res_str = res_str.rstrip(", ")
        res_str += ")"

        return res_str

    @staticmethod
    def repr_instruction_list(obj: Iterable[TPInstruction]) -> str:
        return str([f"{ins.name}:{ins.__class__.__name__}" for ins in obj])

    @staticmethod
    def repr_coupling_df(df: pd.DataFrame, table_name: str) -> str:
        table_str = f"\n\n\t{table_name}:\n\t"
        table_str += "-" * len(table_name) + "\n\t"
        table_str += "\n\t".join(df.to_string().split("\n"))

        return table_str

    @staticmethod
    def repr_list_of_lists(obj: ListWrapper[Iterable[int]]) -> str:
        return str([list(inner) for inner in obj])

    @staticmethod
    def repr_list_wrapper(obj: ListWrapper) -> str:
        if InstructionPrinter.is_instruction_list(obj):
            return f"{InstructionPrinter.repr_instruction_list(obj)}"
        if InstructionPrinter.is_list_of_lists(obj):
            return f"{InstructionPrinter.repr_list_of_lists(obj)}"
        return str(list(obj))

    @staticmethod
    def repr_var_info(
        instruction: TPInstruction, vars_counter: Dict[str, int] = None
    ) -> str:
        hash_var_item: Callable[[str, List[int]], str] = lambda k, v: f"{k}={v}"
        substr = ""

        col_width = 50
        columns = ["Layer", "Output Shape", "Param #"]
        header = "\n\t" + " " * 3
        header += f"{''.join(col.ljust(col_width) for col in columns)}"
        substr += header

        separation_line = "\n\t" + "-" * ((col_width * (len(columns) - 1)) + 10)
        substr += separation_line

        n_params_total = 0
        n_new_vars = 0
        for i, (k, v) in enumerate(instruction.get_variables_info().items()):
            if k == "instruction" or v is None:
                continue
            if vars_counter is not None and hash_var_item(k, v) in vars_counter:
                continue
            n_new_vars += 1
            substr += f"\n\t{i:2d} {k.ljust(col_width, '.')}"
            substr += f"{str(v).ljust(col_width, '.')}"
            n_params = int(np.prod(v))
            substr += f"{str(n_params)}"
            n_params_total += n_params

            if vars_counter is not None:
                vars_counter[hash_var_item(k, v)] = n_params

        substr += separation_line
        substr += f"\n\tTrainable Params: {n_params_total}"
        return substr if n_params_total > 0 else ""

    @staticmethod
    def _format_coupling_cg_list(
        df: pd.DataFrame, format_str: str = ".3f"
    ) -> pd.DataFrame:
        df = df.copy()
        df["cg_list"] = df["cg_list"].apply(
            lambda l: [f"{el:{format_str}}" for el in l]
        )
        return df

    @staticmethod
    def _groupby_and_count_Aterms(df: pd.DataFrame) -> pd.DataFrame:
        return (
            df.query("m==0").groupby(["l", "parity"]).size().reset_index(name="count")
        )


class ParityType(Enum):
    REAL_PARITY = ("REAL_PARITY",)
    PSEUDO_PARITY = ("PSEUDO_PARITY",)
    UNKNOWN_PARITY = ("UNKNOWN_PARITY",)

    @classmethod
    def get_parity_type(cls, parity: List[Tuple[int, int]]) -> "ParityType":
        parity_ar = np.array(parity)

        l_values = parity_ar[:, 0]
        parity_values = parity_ar[:, 1]

        even_mask = l_values % 2 == 0
        odd_mask = l_values % 2 == 1

        if np.all(parity_values[even_mask] == 1) and np.all(
            parity_values[odd_mask] == -1
        ):
            return ParityType.REAL_PARITY
        elif np.all(parity_values[even_mask] == -1) and np.all(
            parity_values[odd_mask] == 1
        ):
            return ParityType.PSEUDO_PARITY
        else:
            return ParityType.UNKNOWN_PARITY
