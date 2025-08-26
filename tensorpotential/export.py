from __future__ import annotations

import copy
import dataclasses
import re

import numpy as np
import yaml

from tensorpotential.instructions.base import TPInstruction, TPEquivariantInstruction
from tensorpotential.poly import init_coupling_symbols, get_symbol, expand_monomial


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


def export_to_yaml(
    fs_ins_dict: list[TPInstruction] | dict[str, TPInstruction], filename: str
):
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

    fs_ins_dict = fs_ins_dict
    if isinstance(fs_ins_dict, list):
        fs_ins_dict = {ins.name: ins for ins in fs_ins_dict}

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

    # assert collector_ins.chemical_embedding is None
    assert collector_ins.is_central_atom_type_dependent
    ndens = collector_ins.n_out

    collector_dict = collector_ins.collector

    A_ins_dict: dict[str, TPEquivariantInstruction] = {
        ins.name: ins for _, ins in fs_ins_dict.items() if ins.name in collector_dict
    }

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
