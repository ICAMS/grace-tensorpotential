from __future__ import annotations

import numpy as np
from typing import Dict, Union, List, Tuple
from collections import defaultdict
import re
import dataclasses
import copy
import yaml


@dataclasses.dataclass
class TDACEBBasisFunction:
    mu0: int
    ns: int  # list  # [rank]
    ls: list[int]  # [rank]
    ms_combs: list[int]  # [num_ms_combs][rank]
    gen_cgs: list[float]  # [num_ms_combs]
    ndensity: int
    coeff: list[float]
    rank: int

    def todict(self):
        res = dataclasses.asdict(self)
        gen_cgs = [float(c) for c in res["gen_cgs"]]
        res["gen_cgs"] = gen_cgs
        return res


def _compute_variables_tuple(variables):
    return tuple(sorted([(k, v) for k, v in variables.items()]))


def _simplify_variable_tuple(variable_tuple_list):
    if len(variable_tuple_list) > 1:
        return tuple((k, v) for k, v in variable_tuple_list if k != "")
    return variable_tuple_list


def _simplify_polynomial(monomials):
    simplified_monomials_dict = defaultdict(int)  # monomial_variable_tuple -> coeff

    for monomial in monomials:
        simplified_monomials_dict[monomial.variables_tuple] += monomial.coefficient

    simplified_monomials = [
        Monomial(coeff, variables_tuple)
        for variables_tuple, coeff in simplified_monomials_dict.items()
        if not np.allclose(coeff, 0)
    ]

    simplified_monomials = sorted(simplified_monomials, key=lambda m: m.variables_tuple)

    return simplified_monomials


class Monomial:
    def __init__(
        self,
        coefficient: Union[int, float],
        variables: Union[Dict[str, int], Tuple, str] = None,
    ):
        self.coefficient = coefficient

        if isinstance(variables, dict):
            self.variables_tuple = _compute_variables_tuple(
                variables
            )  # {"x":2, "y":3} -> ((x,2),(y,3))
        elif isinstance(variables, (list, tuple)):
            self.variables_tuple = tuple(variables)
        elif isinstance(variables, str):
            self.variables_tuple = ((variables, 1),)
        elif variables is None:
            self.variables_tuple = (("", 0),)
        else:
            raise NotImplementedError(f"Unsupported variable: {variables}")

        self.variables_tuple = _simplify_variable_tuple(self.variables_tuple)

    def __eq__(self, other):
        return self.variables_tuple == other.variables_tuple and np.allclose(
            self.coefficient, other.coefficient
        )

    def __add__(
        self, other: Union["Monomial", "Polynomial"]
    ) -> Union["Monomial", "Polynomial", float]:
        if isinstance(other, Monomial):
            if self.variables_tuple == other.variables_tuple:
                coefficient_sum = self.coefficient + other.coefficient
                if np.allclose(coefficient_sum, 0):
                    return 0  # Return zero if the sum is zero
                return Monomial(coefficient_sum, self.variables_tuple)
            else:
                return Polynomial([self, other])
        elif isinstance(other, Polynomial):
            return other + self
        else:
            raise NotImplementedError(
                f"Summation Monomial+{type(other)} is not implemented"
            )

    def __mul__(self, other: "Monomial") -> "Monomial":
        if isinstance(other, (float, int)):
            return Monomial(self.coefficient * other, self.variables_tuple)
        elif isinstance(other, Monomial):
            coefficient = self.coefficient * other.coefficient
            variables_dict = {n: p for n, p in self.variables_tuple}
            for variable, power in other.variables_tuple:
                variables_dict[variable] = variables_dict.get(variable, 0) + power
            return Monomial(coefficient, variables_dict)
        elif isinstance(other, Polynomial):
            return other * self

    def __str__(self) -> str:
        if self.coefficient == 0:
            return "0"
        terms = []
        for variable, power in self.variables_tuple:
            if power == 1:
                terms.append(variable)
            else:
                terms.append(f"{variable}^{power}")
        terms_str = "".join(terms)
        if terms_str:
            if not np.allclose(self.coefficient, 1.0):
                return f"{self.coefficient}*{terms_str}"
            else:
                return f"{terms_str}"
        else:
            return f"{self.coefficient}"

    def __repr__(self):
        return self.__str__()


class Polynomial:
    def __init__(self, monomials: List[Monomial]):
        # simplify list of monomials
        self.monomials = _simplify_polynomial(monomials)

    def __add__(self, other: "Polynomial") -> "Polynomial":
        if isinstance(other, Polynomial):
            monomials = self.monomials + other.monomials
        elif isinstance(other, Monomial):
            monomials = self.monomials + [other]
        elif isinstance(other, (int, float)):
            monomials = self.monomials + [Monomial(other)]
        else:
            raise NotImplementedError(
                f"Summation Polynomial+{type(other)} is not implemented"
            )
        return Polynomial(monomials)

    def __mul__(self, other) -> "Polynomial":
        if isinstance(other, Polynomial):
            other_monomials = other.monomials
        elif isinstance(other, Monomial):
            other_monomials = [other]
        elif isinstance(other, (int, float)):
            other_monomials = [Monomial(other)]
        else:
            raise NotImplementedError(
                f"Multiplication Polynomial*{type(other)} is not implemented"
            )

        monomials = []
        for monomial1 in self.monomials:
            for monomial2 in other_monomials:
                monomials.append(monomial1 * monomial2)
        return Polynomial(monomials)

    def __eq__(self, other):
        if len(self.monomials) != len(other.monomials):
            return False
        return all([a == b for (a, b) in zip(self.monomials, other.monomials)])

    def __str__(self) -> str:
        if not self.monomials:
            return "0"
        return " + ".join(monomial.__str__() for monomial in self.monomials)

    def __repr__(self):
        return self.__str__()


def init_coupling_symbols(instruction):
    cmd = instruction.coupling_meta_data
    cmd["symbol"] = cmd.apply(
        lambda row: Polynomial(
            [Monomial(1.0, f"{instruction.name}_({row['l']},{row['m']})")]
        ),
        axis=1,
    )


def get_symbol(instruction_name, coupling_index, instructions_dict):
    coupling_instruction = instructions_dict[instruction_name]

    coupling_meta_df = coupling_instruction.coupling_meta_data

    if "symbol" not in coupling_meta_df.columns:
        coupling_meta_df["symbol"] = None

    cur_symbol = coupling_meta_df.loc[coupling_index, "symbol"]

    if cur_symbol is not None:
        return cur_symbol
    else:
        left_instruction_name, right_instruction_name = (
            coupling_instruction.left.name,
            coupling_instruction.right.name,
        )

        row = coupling_meta_df.loc[coupling_index]

        cur_symbol = Polynomial([Monomial(0)])
        for lind, rind, cg in zip(row["left_inds"], row["right_inds"], row["cg_list"]):
            cur_symbol += (
                get_symbol(left_instruction_name, lind, instructions_dict)
                * get_symbol(right_instruction_name, rind, instructions_dict)
                * cg
            )
        coupling_meta_df.loc[coupling_index, "symbol"] = cur_symbol

        return cur_symbol


def expand_monomial(mon: Monomial):
    expanded_monomial_list = []
    for name, power in mon.variables_tuple:
        for _ in range(power):
            expanded_monomial_list.append(name)

    return tuple(expanded_monomial_list), mon.coefficient


def poly_norm(poly, order=2):
    return sum([abs(mon.coefficient) ** 2 for mon in poly.monomials])


def normalize_poly(poly, order=2):
    if poly is not None:
        norm = poly_norm(poly, order=order)

        for mon in poly.monomials:
            mon.coefficient /= norm ** (1 / order)
        return poly


def extract_const_shift_scale(fs_ins_dict):
    if "ConstantScaleShiftTarget" in fs_ins_dict:
        const_shift_scale = fs_ins_dict["ConstantScaleShiftTarget"]

        scale = const_shift_scale.scale
        try:
            scale = scale.numpy()
        except:
            pass

        constant_shift = const_shift_scale.constant_shift

        if not isinstance(constant_shift, (float, int)):
            constant_shift = constant_shift.numpy()

        atomic_shift_map = const_shift_scale.atomic_shift_map
        try:
            if atomic_shift_map is not None:
                atomic_shift_map = atomic_shift_map.numpy().flatten()
        except Exception:
            pass

        return scale, constant_shift, atomic_shift_map
    else:
        return None


def export_to_yaml(fs_ins: List, filename: str):
    # to avoid circular import
    from tensorpotential.instructions.compute import (
        RadialBasis,
        LinearRadialFunction,
        ScalarChemicalEmbedding,
        SingleParticleBasisFunctionScalarInd,
        FunctionReduce,
        FunctionReduceN,
    )
    from tensorpotential.instructions.output import FSOut2ScalarTarget

    fs_ins_dict = {ins.name: ins for ins in fs_ins}

    rad_basis = fs_ins_dict["RadialBasis"]
    assert isinstance(rad_basis, RadialBasis)

    chem_emb_fun = fs_ins_dict["Z"]
    assert isinstance(chem_emb_fun, ScalarChemicalEmbedding)

    sing_part_basis_fun = fs_ins_dict["A"]
    assert isinstance(sing_part_basis_fun, SingleParticleBasisFunctionScalarInd)

    lin_rad_fun = fs_ins_dict["R"]
    assert isinstance(lin_rad_fun, LinearRadialFunction)

    collector_ins = fs_ins_dict["E"]
    assert isinstance(collector_ins, (FunctionReduce, FunctionReduceN))

    fs_emb_ins = fs_ins_dict["FSOut2ScalarTarget"]
    assert isinstance(fs_emb_ins, FSOut2ScalarTarget)

    elements = [s.decode() for s in chem_emb_fun.element_map_symbols.numpy()]

    rad_basis_type = rad_basis.basis_type
    nradbase = rad_basis.basis_function.nfunc
    cutoff = rad_basis.basis_function.rcut
    nradmax = lin_rad_fun.n_rad_max

    assert collector_ins.chemical_embedding is None
    assert collector_ins.is_central_atom_type_dependent
    ndens = collector_ins.n_out

    collector_dict = collector_ins.collector

    A_ins_dict = {ins.name: ins for ins in fs_ins if ins.name in collector_dict}

    A_ins = A_ins_dict["A"]
    init_coupling_symbols(A_ins)

    coeffs_list = []

    # mult by collector_ins.norm_{k}
    for ci in collector_ins.instructions:
        norm_coeff = getattr(collector_ins, f"norm_{ci.name}").numpy()
        coeffs_list.append(
            getattr(collector_ins, f"reducing_{ci.name}").numpy() * norm_coeff
        )  # list of arrays (shape=[num_elems, num_dens, nrad, num_ls_0_funcs])

    # actually collected B-funcs
    collected_polynomials = []
    for name, val in collector_dict.items():
        col_index = val["func_collect_ind"]
        for ind in col_index:
            collected_polynomials.append([get_symbol(name, ind, A_ins_dict), name, ind])

    p = re.compile("([0-9-]+)")

    mu0 = 0

    ls_b_funcs_list = []
    for pol, name, ind in collected_polynomials:

        eml = [expand_monomial(mon) for mon in pol.monomials]

        total_ls = set()
        total_ms = []
        gen_cgs = []
        for mons, cg in eml:
            ls = []
            ms = []
            for mon in mons:
                l, m = map(int, p.findall(mon))
                ls.append(l)
                ms.append(m)
            total_ls.add(tuple(ls))
            total_ms += ms
            gen_cgs.append(cg)

        total_ls = list(total_ls)

        assert len(total_ls) == 1

        total_ls = list(total_ls[0])

        bfunc = TDACEBBasisFunction(
            rank=len(total_ls),
            ms_combs=total_ms,  # [num_ms_combs][rank]
            ns=None,  # [rank]
            ls=total_ls,  # [rank]
            ndensity=ndens,
            mu0=mu0,
            gen_cgs=list(gen_cgs),
            coeff=0.0,
        )

        ls_b_funcs_list.append(bfunc)

    num_of_ls_funcs = sum([c.shape[-1] for c in coeffs_list])
    assert num_of_ls_funcs == len(
        ls_b_funcs_list
    ), "Num. of coefficients and num. of basis functions do not match"

    functions_dict = {}

    for el_ind, el in enumerate(elements):
        el_coeffs_list = [c[el_ind] for c in coeffs_list]
        functions_dict[el_ind] = []

        for n in range(nradmax):
            cur_coeffs_list = [
                el_coeffs[:, n, :].T
                for el_coeffs in el_coeffs_list
                if el_coeffs.shape[1] > n
            ]

            cur_coeffs = np.concatenate(cur_coeffs_list)

            for orig_bfunc, cur_coeff in zip(ls_b_funcs_list, cur_coeffs):
                bfunc = copy.copy(orig_bfunc)

                # rank = bfunc.rank

                bfunc.ns = n + 1  # [n + 1] * rank
                bfunc.ndensity = ndens
                bfunc.mu0 = el_ind
                bfunc.coeff = cur_coeff.tolist()

                functions_dict[el_ind].append(bfunc)

    functions_dict = {
        k: sorted(v, key=lambda f: (f.rank, f.ns, f.ls))
        for k, v in functions_dict.items()
    }

    crad = lin_rad_fun.crad.numpy()  # shape = [n_rad_max, lmax + 1, n_rad_base]
    chem_emb = chem_emb_fun.w.numpy()  # shape = [n_elem, chem_emb_size]
    Z_to_n = sing_part_basis_fun.lin_transform.w.numpy()
    Z_norm = sing_part_basis_fun.lin_transform.norm.numpy()
    chem_Z = np.einsum("ek,kn->en", chem_emb, Z_to_n) * Z_norm

    rad_basis_spec = {
        "n_rad_base": nradbase,
        "nradmax": nradmax,
        "radbasename": rad_basis_type,
        "cutoff": cutoff,
        "crad": crad.flatten().tolist(),
        "Z": chem_Z.flatten().tolist(),
        "crad_shape": list(np.shape(crad)),
        "Z_shape": list(np.shape(chem_Z)),
    }

    emb_spec = {
        "type": "FinnisSinclairShiftedScaled",
        "params": np.array(fs_emb_ins.fs_parameters).flatten().tolist(),
    }

    tdyaml_dict = {
        "elements": elements,
        "nnorm": float(A_ins.inv_avg_n_neigh.numpy()),
        "emb_spec": emb_spec,
        "radial_basis": rad_basis_spec,
        "functions": {k: [f.todict() for f in v] for k, v in functions_dict.items()},
    }

    shift_scale_tuple = extract_const_shift_scale(fs_ins_dict)
    if shift_scale_tuple is not None:
        scale, constant_shift, atomic_shift_map = shift_scale_tuple
        if atomic_shift_map is not None:
            tdyaml_dict["E0"] = atomic_shift_map.tolist()
        if constant_shift is not None:
            tdyaml_dict["shift"] = float(constant_shift)
        if scale is not None:
            tdyaml_dict["scale"] = float(scale)

    # Step 3. Store potential in .tdyaml

    # custom save yaml
    with open(filename, "w+") as f:
        for key, yaml_obj in tdyaml_dict.items():
            f.write(key + ": ")
            if key != "functions":
                f.write(
                    yaml.dump(
                        yaml_obj,
                        Dumper=yaml.CDumper,
                        default_flow_style=True,
                    )
                )
            else:
                f.write("\n")
                for kk, funcs in yaml_obj.items():
                    f.write("    " + str(kk) + ": \n")
                    for func in funcs:
                        f.write(
                            "     - "
                            + yaml.dump(
                                func,
                                Dumper=yaml.CDumper,
                                default_flow_style=True,
                                sort_keys=False,
                                width=999999,
                            )
                        )
