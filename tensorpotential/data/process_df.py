from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd


from collections import Counter, defaultdict
from scipy.spatial import ConvexHull

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(seq, *args, **kwargs):
        return seq


ASE_ATOMS = "ase_atoms"
COMP_DICT = "comp_dict"
COMP_TUPLE = "comp_tuple"
REFERENCE_ENERGY = "reference_energy"
ENERGY = "energy"
NUMBER_OF_ATOMS = "NUMBER_OF_ATOMS"
E_CORRECTED_PER_ATOM_COLUMN = "energy_corrected_per_atom"
WEIGHTS_FACTOR = "w_factor"
REF_ENERGY_KW = "ref_energy"
E_CHULL_DIST_PER_ATOM = "e_chull_dist_per_atom"
E_FORMATION_PER_ATOM = "e_formation_per_atom"
EFFECTIVE_ENERGY = "effective_energy"
ENERGY_CORRECTED_COL = "energy_corrected"
FORCES_COL = "forces"

SINGLE_ATOM_ENERGY_DICT = {
    # energy computed with default VASP_NM settings (500eV, Gaussian smearing, sigma=0.1, min 10x10x10 A cell, single atom, NM)
    "VASP_PBE_500_0.125_0.1_NM": {
        "Ac": -0.20899015,
        "Ag": -0.06689862,
        "Al": -0.10977253,
        "Ar": -0.0235748,
        "As": -0.17915798,
        "Au": -0.06427594,
        "B": -0.1100114,
        "Ba": -0.13720005,
        "Be": -0.04039112,
        "Bi": -0.1854587,
        "Br": -0.12188631,
        "C": -0.1612024,
        "Ca": -0.02585929,
        "Cd": -0.01423875,
        "Ce": -0.89994905,
        "Cl": -0.12414702,
        "Co": -0.27078113,
        "Cr": -0.65569845,
        "Cs": -0.23242861,
        "Cu": -0.06659418,
        "Dy": -0.15964479,
        "Er": -0.1606716,
        "Eu": -0.05150557,
        "F": -0.11863189,
        "Fe": -0.35795312,
        "Fr": -0.20499698,
        "Ga": -0.10956489,
        "Gd": -0.16319873,
        "Ge": -0.16372306,
        "H": -0.05660531,
        "He": 0.00161163,
        "Hf": -2.79098326,
        "Hg": -0.01069226,
        "Ho": -0.15904603,
        "I": -0.11835143,
        "In": -0.12375132,
        "Ir": -0.27663991,
        "K": -0.07961456,
        "Kr": -0.02228048,
        "La": -0.40271396,
        "Li": -0.06085926,
        "Lu": -0.16197678,
        "Mg": -0.00056567,
        "Mn": -0.47998212,
        "Mo": -0.4297273,
        "N": -0.17273379,
        "Na": -0.06438759,
        "Nb": -0.82269379,
        "Nd": -0.18393967,
        "Ne": -0.01244429,
        "Ni": -0.18976919,
        "Np": -4.40174385,
        "O": -0.16665686,
        "Os": -0.60389947,
        "P": -0.18102125,
        "Pa": -1.40133833,
        "Pb": -0.16754695,
        "Pd": -1.47571017,
        "Pm": -0.17541143,
        "Po": -0.17215562,
        "Pt": -0.25488222,
        "Pu": -6.19533094,
        "Ra": -0.10539559,
        "Rb": -0.07907932,
        "Re": -1.20230599,
        "Rh": -1.03428996,
        "Rn": -0.00530062,
        "Ru": -0.60938402,
        "S": -0.16942446,
        "Sb": -0.17675287,
        "Sc": -1.82317913,
        "Se": -0.1669939,
        "Si": -0.16208649,
        "Sm": -0.17184578,
        "Sn": -0.17573341,
        "Sr": -0.02800043,
        "Ta": -2.3423585,
        "Tb": -0.16170736,
        "Tc": -0.36811991,
        "Te": -0.16457883,
        "Th": -0.65883239,
        "Ti": -1.35069169,
        "Tl": -0.11645191,
        "Tm": -0.16111118,
        "U": -2.73254914,
        "V": -0.93934306,
        "W": -1.81667003,
        "Xe": -0.01207016,
        "Y": -1.98890231,
        "Yb": -0.02974997,
        "Zn": -0.0113131,
        "Zr": -1.43289387,
    }
}


ELEMENTAL_REF_ENERGIES_DICT = defaultdict(lambda: 0.0)
ELEMENTAL_REF_ENERGIES_DICT.update(
    {
        # 2023-02-07-mp-elemental-reference-entries.json
        "Ne": -0.02593678,
        "He": -0.00905951,
        "Ar": -0.06880822,
        "F": -1.9114789675,
        "O": -4.94668871125,
        "Cl": -1.84853666,
        "N": -8.336494925,
        "Kr": -0.05671467,
        "Br": -1.55302833,
        "I": -1.47336635,
        "Xe": -0.03617417,
        "S": -4.136449866875,
        "Se": -3.49591147765625,
        "C": -9.2286654925,
        "Au": -3.273882,
        "W": -12.95813023,
        "Pb": -3.71264707,
        "Rh": -7.36430787,
        "Pt": -6.07113332,
        "Ru": -9.27440254,
        "Pd": -5.17988181,
        "Os": -11.22736743,
        "Ir": -8.83843418,
        "H": -3.392726045,
        "P": -5.413302506666667,
        "As": -4.659118405,
        "Mo": -10.84565011,
        "Te": -3.1433058933333338,
        "Sb": -4.12900124,
        "B": -6.679391770833334,
        "Bi": -3.84048913,
        "Ge": -4.623027855,
        "Hg": -0.303680365,
        "Sn": -4.009571855,
        "Ag": -2.8325560033333335,
        "Ni": -5.78013668,
        "Tc": -10.360638945,
        "Si": -5.42531803,
        "Re": -12.444527185,
        "Cu": -4.09920667,
        "Co": -7.108317795,
        "Fe": -8.47002121,
        "Ga": -3.0280960225,
        "In": -2.75168373,
        "Cd": -0.92288976,
        "Cr": -9.65304747,
        "Zn": -1.2597436100000001,
        "V": -9.08390607,
        "Tl": -2.3626431466666666,
        "Al": -3.74557583,
        "Nb": -10.10130504,
        "Be": -3.739412865,
        "Mn": -9.162015292068965,
        "Ti": -7.895492016666666,
        "Ta": -11.85777763,
        "Pa": -9.51466466,
        "U": -11.29141001,
        "Sc": -6.332469105,
        "Np": -12.94777968125,
        "Zr": -8.54770063,
        "Mg": -1.60028005,
        "Th": -7.41385825,
        "Hf": -9.95718903,
        "Pu": -14.26783833,
        "Lu": -4.52095052,
        "Tm": -4.475835423333334,
        "Er": -4.56771881,
        "Ho": -4.58240887,
        "Y": -6.466471113333333,
        "Dy": -4.60678684,
        "Gd": -14.07612224,
        "Eu": -10.2570018,
        "Sm": -4.718586135,
        "Nd": -4.7681474325,
        "Pr": -4.780905755,
        "Pm": -4.7505423225,
        "Ce": -5.933089155,
        "Yb": -1.5396082800000002,
        "Tb": -4.6343661,
        "La": -4.936007105,
        "Ac": -4.1211750075,
        "Ca": -2.00559988,
        "Li": -1.9089228666666667,
        "Sr": -1.6894934533333332,
        "Na": -1.3225252934482759,
        "Ba": -1.91897494,
        "Rb": -0.9805340725,
        "K": -1.110398947,
        "Cs": -0.8954023720689656,
        # cohesive energies from https://www.knowledgedoor.com/2/elements_handbook/cohesive_energy.html
        "Po": -1.50,
        "Rn": -0.202,
        "Ra": -1.66,
        "Am": -2.73,
        "Cm": -3.99,
    }
)


def compdict_to_comptuple(comp_dict):
    n_atoms = sum([v for v in comp_dict.values()])
    return tuple(sorted([(k, v / n_atoms) for k, v in comp_dict.items()]))


def comptuple_to_str(comp_tuple):
    return " ".join(("{}_{:.3f}".format(e, c) for e, c in comp_tuple))


def compute_compositions(
    df: pd.DataFrame, ase_atoms_column=ASE_ATOMS, compute_composition_tuples=True
):
    """
    Generate new columns:
       'comp_dict' - composition dictionary
       'n_atom' - number of atoms
       'n_'+Elements, 'c_'+Elements - number and concentration of elements
    """
    df[COMP_DICT] = df[ase_atoms_column].map(
        lambda atoms: Counter(atoms.get_chemical_symbols())
    )
    df[NUMBER_OF_ATOMS] = df[ase_atoms_column].map(len)

    if compute_composition_tuples:
        df[COMP_TUPLE] = df[COMP_DICT].map(compdict_to_comptuple)

    elements = extract_elements(df)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
        for el in elements:
            df["n_" + el] = df[COMP_DICT].map(lambda d: d.get(el, 0))
            df["c_" + el] = df["n_" + el] / df[NUMBER_OF_ATOMS]
    return elements


def extract_elements(df: pd.DataFrame, composition_dict_column=COMP_DICT):
    elements_set = set()
    for cd in df[composition_dict_column]:
        elements_set.update(cd.keys())
    elements = sorted(elements_set)
    return elements


def compute_formation_energy(
    df: pd.DataFrame,
    elements=None,
    epa_gs_dict=None,
    energy_per_atom_column="energy_per_atom",
    verbose=True,
):
    """
    Compute formation energies in `df` using  `energy_per_atom_column` ("energy_per_atom" by default)

    Parameters:
        df: pd.DataFrame
        elements: list of elements, If None - extract automatically
        epa_gs_dict: dict of reference (ground states) energies per element to calculate formation energy
                    or "mp_elemental_ref_entries"
        energy_per_atom_column: str - name of the column in `df` that contains energies per atom
        verbose: bool

    Returns: None (all modifications are done in `df` in-place)
    """
    if elements is None:
        elements = extract_elements(df)

    c_elements = ["c_" + el for el in elements]

    if epa_gs_dict is None:
        epa_gs_dict = {}
        for el in elements:
            subdf = df[df["c_" + el] == 1.0]
            if len(subdf) > 0:
                e_min_pa = subdf[energy_per_atom_column].min()
            else:
                e_min_pa = 0.0
                if verbose:
                    print(
                        "No pure element energy for {} is available, assuming 0  eV/atom".format(
                            el
                        )
                    )
            epa_gs_dict[el] = e_min_pa
    elif epa_gs_dict == "mp_elemental_ref_entries":
        epa_gs_dict = ELEMENTAL_REF_ENERGIES_DICT
    element_emin_array = np.array([epa_gs_dict[el] for el in elements])
    c_conc = df[c_elements].values
    e_formation_ideal = np.dot(c_conc, element_emin_array)
    df[E_FORMATION_PER_ATOM] = df[energy_per_atom_column] - e_formation_ideal


# TODO: write tests
def compute_convexhull_dist(
    df: pd.DataFrame,
    ase_atoms_column=ASE_ATOMS,
    energy_per_atom_column="energy_per_atom",
    verbose=True,
    max_allowed_elements=6,
):
    """
    df: pd.DataFrame with ASE atoms column and energy-per-atom column
    ase_atoms_column: (str) name of ASE atoms column
    energy_per_atom_column: (str) name of energy-per-atom column

    return: list of elements (str)

    construct new columns to dataframe:
     'comp_dict': composition dictionary
     'n_'+element, 'c_'+element - number and concentration of elements
     'e_formation_per_atom': formation energy per atom
     'e_chull_dist_per_atom': distance to convex hull
    """
    elements = compute_compositions(df, ase_atoms_column=ase_atoms_column)
    c_elements = ["c_" + el for el in elements]
    if max_allowed_elements is not None and len(c_elements) > max_allowed_elements:
        raise RuntimeError(
            f"Number of elements {len(c_elements)} more than {max_allowed_elements}, construction of convex hull is unmanageable."
        )

    compute_formation_energy(
        df, elements, energy_per_atom_column=energy_per_atom_column, verbose=verbose
    )

    # check if more than one unique compositions
    uniq_compositions = df[COMP_TUPLE].unique()
    # df.drop(columns=["comp_tuple"], inplace=True)

    if len(uniq_compositions) > 1:
        if verbose:
            logging.warning(
                "Structure dataset: multiple unique compositions found, trying to construct convex hull"
            )
        chull_values = df[c_elements[:-1] + [E_FORMATION_PER_ATOM]].values
        hull = ConvexHull(chull_values)
        ok = hull.equations[:, -2] < 0
        selected_simplices = hull.simplices[ok]
        selected_equations = hull.equations[ok]

        norms = selected_equations[:, :-1]
        offsets = selected_equations[:, -1]

        norms_c = norms[:, :-1]
        norms_e = norms[:, -1]

        e_chull_dist_list = []
        for p in chull_values:
            p_c = p[:-1]
            p_e = p[-1]
            e_simplex_projections = []
            for nc, ne, b, simplex in zip(
                norms_c, norms_e, offsets, selected_simplices
            ):
                if ne != 0:
                    e_simplex = (-b - np.dot(nc, p_c)) / ne
                    e_simplex_projections.append(e_simplex)
                elif (
                    np.abs(b + np.dot(nc, p_c)) < 2e-15
                ):  # ne*e_simplex + b + np.dot(nc,p_c), ne==0
                    e_simplex = p_e
                    e_simplex_projections.append(e_simplex)

            e_simplex_projections = np.array(e_simplex_projections)

            mask = e_simplex_projections < p_e + 1e-15

            e_simplex_projections = e_simplex_projections[mask]

            e_dist_to_chull = np.min(p_e - e_simplex_projections)

            e_chull_dist_list.append(e_dist_to_chull)

        e_chull_dist_list = np.array(e_chull_dist_list)
    else:
        if verbose:
            logging.info(
                "Structure dataset: only single unique composition found, switching to cohesive energy reference"
            )
        emin = df[energy_per_atom_column].min()
        e_chull_dist_list = df[energy_per_atom_column] - emin

    df[E_CHULL_DIST_PER_ATOM] = e_chull_dist_list
    return elements


def compute_corrected_energy(
    df: pd.DataFrame,
    esa_dict=None,
    calculator_name="VASP_PBE_500_0.125_0.1_NM",
    n_atoms_column=NUMBER_OF_ATOMS,
):
    elements = compute_compositions(df)
    n_elements = ["n_" + e for e in elements]
    if esa_dict is None:
        esa_dict = {e: SINGLE_ATOM_ENERGY_DICT[calculator_name][e] for e in elements}
    esa_array = np.array([esa_dict.get(e, 0) for e in elements])
    corr_mask = ~df[ENERGY].isna()
    df.loc[corr_mask, ENERGY_CORRECTED_COL] = df.loc[corr_mask, ENERGY] - (
        df.loc[corr_mask, n_elements] * esa_array
    ).sum(axis=1)
    e_corr_shift = esa_dict.get("shift", 0)
    df[ENERGY_CORRECTED_COL] += e_corr_shift * df[n_atoms_column]
    df[E_CORRECTED_PER_ATOM_COLUMN] = df[ENERGY_CORRECTED_COL] / df[n_atoms_column]
    return esa_dict


#
# def compute_shifted_scaled_corrected_energy(
#     df: pd.DataFrame, n_atoms_column=NUMBER_OF_ATOMS
# ):
#     elements, df = compute_compositions(df)
#     n_elements = ["n_" + e for e in elements]
#     corr_mask = ~df[ENERGY].isna()
#     comp_array = df.loc[corr_mask, n_elements].values
#     e_array = df.loc[corr_mask, ENERGY].values
#     assert not np.any(np.isnan(comp_array)), "Compositions columns contain NaN"
#     assert not np.any(np.isnan(e_array)), "Energy column contain NaN"
#     esa_array = np.linalg.pinv(comp_array, rcond=1e-10) @ e_array  # solve equation
#     esa_dict = {e: esa for e, esa in zip(elements, esa_array)}
#     df.loc[corr_mask, ENERGY_CORRECTED_COL] = e_array - np.dot(comp_array, esa_array)
#     df[E_CORRECTED_PER_ATOM_COLUMN] = df[ENERGY_CORRECTED_COL] / df[n_atoms_column]
#
#     def safe_get_volume_per_atom(at):
#         try:
#             return at.get_volume() / len(at)
#         except Exception as e:
#             return 0
#
#     df["volume_per_atom"] = df[ASE_ATOMS].map(safe_get_volume_per_atom)
#     max_vpa = df["volume_per_atom"].max()
#     e_corr_shift = 0
#     if max_vpa > 0:
#         # "-", becese +=shift
#         e_corr_shift = -df[df["volume_per_atom"] >= max_vpa].iloc[0][
#             E_CORRECTED_PER_ATOM_COLUMN
#         ]
#     else:  # max_vpa==0
#         e_corr_shift = 0
#
#     df[ENERGY_CORRECTED_COL] += e_corr_shift * df[n_atoms_column]
#     esa_dict["shift"] = e_corr_shift
#     return esa_dict, df
