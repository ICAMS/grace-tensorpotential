from __future__ import annotations

from collections import defaultdict
from typing import Union, Dict, Tuple, List

import numpy as np


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
