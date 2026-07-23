# ==============================================================================
# Imports
# ==============================================================================
from __future__ import annotations

import os
from typing import Callable

from networkx import MultiDiGraph

from rFBApy.bn.bexpr import BoolExpr, parse


# ==============================================================================
# Regulatory Network
# ==============================================================================
class RegulatoryNetwork:
    # ==========================================================================
    # Initialization
    # ==========================================================================
    def __init__(self: RegulatoryNetwork):
        self.__rules: dict[str, BoolExpr] = {}
        self.__fn: dict[
            str, Callable[[dict[str, bool | int | float]], bool | None]
        ] = {}

        self.__components: frozenset[str] | None = None
        self.__constants: frozenset[str] | None = None
        self.__thresholds: frozenset[tuple[str, float]] | None = None

    # ==========================================================================
    # Properties
    # ==========================================================================
    @property
    def nodes(self: RegulatoryNetwork) -> frozenset[str]:
        return frozenset(self.__rules)

    @property
    def components(self: RegulatoryNetwork) -> frozenset[str]:
        if self.__components is None:
            self.__components = frozenset(self.__rules.keys()).union(
                *(rule.variables for rule in self.__rules.values())
            )
        return self.__components

    @property
    def influence_graph(self: RegulatoryNetwork) -> MultiDiGraph:
        g: MultiDiGraph = MultiDiGraph()
        for n, rule in self.__rules.items():
            signed_vars: frozenset[tuple[str, int]] = rule.signed_variables
            for u, s in signed_vars:
                g.add_edge(u, n, sign=s)
        return g

    @property
    def undefined(self: RegulatoryNetwork) -> frozenset[str]:
        return self.components.difference(self.nodes)

    @property
    def constants(self: RegulatoryNetwork) -> frozenset[str]:
        if self.__constants is None:
            self.__constants = frozenset(
                [n for n, rule in self.__rules.items() if rule.is_constant()]
            )
        return self.__constants

    @property
    def thresholds(self: RegulatoryNetwork) -> frozenset[tuple[str, float]]:
        if self.__thresholds is None:
            self.__thresholds = frozenset().union(
                *(rule.thresholds for rule in self.__rules.values())
            )
        return self.__thresholds

    # ==========================================================================
    # Getters / Setters
    # ==========================================================================
    def __iter__(self: RegulatoryNetwork):
        yield from self.__rules

    def __getitem__(self: RegulatoryNetwork, node: str) -> BoolExpr:
        return self.__rules[node]

    def __setitem__(
        self: RegulatoryNetwork,
        node: str,
        rule: str | int | BoolExpr,
    ) -> None:
        if isinstance(rule, int):
            rule = str(rule)
        self.__rules[node] = rule if isinstance(rule, BoolExpr) else parse(rule)
        self.__fn[node] = self.__rules[node].simplify().compile()
        # ~ Reset all memoized elements
        self.__thresholds = None
        self.__constants = None
        self.__components = None

    def __delitem__(self: RegulatoryNetwork, node: str):
        del self.__rules[node]
        del self.__fn[node]
        self.__thresholds = None
        self.__constants = None
        self.__components = None

    # ==========================================================================
    # Evaluation
    # ==========================================================================
    def __call__(
        self: RegulatoryNetwork,
        context: dict[str, int | float | bool],
    ) -> dict[str, int | None]:
        vals: dict[str, bool | None] = {
            n: fn(context) for n, fn in self.__fn.items()
        } 
        return {
            n: None if v is None else (1 if v else 0) for n, v in vals.items()
        }

    # ==========================================================================
    # Display
    # ==========================================================================
    def __str__(self: RegulatoryNetwork) -> str:
        return "\n".join(f"{n} <- {rule}" for n, rule in self.__rules.items())

    def __repr__(self: RegulatoryNetwork) -> str:
        return (
            f"{type(self).__name__}(" + ", ".join(
                f"Rule({n};{repr(rule)}"
                for n, rule in self.__rules.items()
            ) + ")"
        )

    # ==========================================================================
    # Parsing
    # ==========================================================================
    # --------------------------------------------------------------------------
    # BNET
    # --------------------------------------------------------------------------
    @staticmethod
    def load_bnet(bnet: str) -> RegulatoryNetwork:
        if not os.path.isfile(bnet):
            raise FileNotFoundError(
                f"Regulatory network BNET file not found: {bnet}"
            )
        # ~ Initialize output --------------------------------------------------
        rn = RegulatoryNetwork()
        # ~ Read BNET file -----------------------------------------------------
        with open(bnet, "r", encoding="utf-8") as file:
            for line in file.readlines():
                line: str = line.split("#")[0].strip()  # remove comments
                if len(line) == 0:
                    continue
                node, rule_str = [e.strip() for e in line.split(",")[:2]]
                rn[node] = rule_str
        # ~ Output -------------------------------------------------------------
        return rn

    # --------------------------------------------------------------------------
    # SBML-Qual
    # --------------------------------------------------------------------------
    @staticmethod
    def load_sbml(sbml: str) -> RegulatoryNetwork:
        raise NotImplementedError()

    # ==========================================================================
    # Export
    # ==========================================================================
    def to_bnet(self: RegulatoryNetwork, filename: str) -> None:
        with open(filename, "w", encoding="utf-8") as file:
            for n, rule in self.__rules.items():
                file.write(f"{n}, {str(rule)}\n")

# ==============================================================================
# Main
# ==============================================================================
if __name__ == "__main__":
    ...
