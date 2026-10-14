# ==============================================================================
# Imports
# ==============================================================================
from __future__ import annotations

import json
from decimal import Decimal
from fractions import Fraction
from math import isinf
from pathlib import Path
from typing import TYPE_CHECKING

import clingo

# ~ Type hints only: both networks import this module
if TYPE_CHECKING:
    from rFBApy.metabolic_network import MetabolicNetwork
    from rFBApy.regulatory_network import RegulatoryNetwork

# ==============================================================================
# ASP terms
# ==============================================================================
_MAX_INT: int = 2**31 - 1


def asp_string(name: str) -> str:
    return json.dumps(str(name))


def asp_number(value: float) -> str:
    value = float(value)
    if isinf(value):
        return '"inf"' if value > 0 else '"-inf"'
    if value.is_integer() and abs(value) <= _MAX_INT:
        return str(int(value))
    # ~ Plain decimal notation, never an exponent: "0.000000004", not "4e-09"
    text: str = format(Decimal(repr(value)), "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return f'"{text}"'


def asp_value(term: clingo.Symbol) -> Fraction | float:
    # ~ Inverse of asp_number, exact except for "inf"
    if term.type is clingo.SymbolType.Number:
        return Fraction(term.number)
    return float(term.string) if "inf" in term.string else Fraction(term.string)


def write_asp(lines: list[str], filename: str | None) -> str:
    asp: str = "\n".join(lines) + "\n"
    if filename is not None:
        with open(filename, "w", encoding="utf-8") as file:
            file.write(asp)
    return asp


# ==============================================================================
# ASP instance
# ==============================================================================
def medium_to_asp(medium: dict[str, float]) -> str:
    # ~ A metabolite at 0 gets no fact: missing, so positive iff produced
    return "".join(
        f"medium({asp_string(metabolite)},{asp_number(concentration)}).\n"
        for metabolite, concentration in medium.items()
        if float(concentration) != 0
    )


def state_to_asp(state: dict[str, int]) -> str:
    return "".join(
        f"x({asp_string(node)}).\n" if value else f":- x({asp_string(node)}).\n"
        for node, value in state.items()
    )


def instance_to_asp(
    mn: MetabolicNetwork,
    bn: RegulatoryNetwork,
    obj: str,
    medium: dict[str, float] = {},  # noqa: B006
    state: dict[str, int] = {},  # noqa: B006
    filename: str | None = None,
) -> str:
    if obj not in mn.reactions():
        raise ValueError(f"Objective reaction '{obj}' not found in the metabolic network.")
    instance = "".join([
        bn.to_asp(),
        mn.to_asp(),
        "% Medium\n",
        medium_to_asp(medium),
        "% Objective\n",
        f"obj({asp_string(obj)}).\n",
        "% Enforced state\n",
        state_to_asp(state),
    ])
    if filename is not None:
        Path(filename).write_text(instance, encoding="utf-8")
    return instance


# ==============================================================================
# Grounding functions
# ==============================================================================
# > @sign, @cmp and @clash of fixpoint/asp/fixpoint.lp, given to
# > clingo.Control.ground as its context: exact comparisons of numbers that
# > ASP cannot order ("0.21", "inf").
# ==============================================================================
class GroundingContext:
    @staticmethod
    def sign(term: clingo.Symbol) -> clingo.Symbol:
        value = asp_value(term)
        return clingo.Number((value > 0) - (value < 0))

    @staticmethod
    def cmp(op: clingo.Symbol, left: clingo.Symbol, right: clingo.Symbol) -> clingo.Symbol:
        # ~ 1 if `left op right`, for op in {gt, lt}
        first, second = asp_value(left), asp_value(right)
        return clingo.Number(int(first > second if op.name == "gt" else first < second))

    @staticmethod
    def clash(
        op1: clingo.Symbol,
        bound1: clingo.Symbol,
        sign1: clingo.Symbol,
        op2: clingo.Symbol,
        bound2: clingo.Symbol,
        sign2: clingo.Symbol,
    ) -> clingo.Symbol:
        # ~ 1 if no value gives [A op1 bound1] the truth value sign1 and
        # ~ [A op2 bound2] the truth value sign2
        def half(op: clingo.Symbol, bound: clingo.Symbol, sign: clingo.Symbol):  # noqa: ANN202
            # ~ Half-line where [A op bound] has value sign: (lower bound?, bound, closed?)
            holds = sign.number == 1
            return (op.name == "gt") == holds, asp_value(bound), not holds

        first, second = half(op1, bound1, sign1), half(op2, bound2, sign2)
        if first[0] == second[0]:
            return clingo.Number(0)
        (_, lower, lower_closed), (_, upper, upper_closed) = (
            (first, second) if first[0] else (second, first)
        )
        empty = lower > upper or (lower == upper and not (lower_closed and upper_closed))
        return clingo.Number(int(empty))
