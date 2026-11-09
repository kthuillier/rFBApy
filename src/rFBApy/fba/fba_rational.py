# ==============================================================================
# Imports
# ==============================================================================
from __future__ import annotations

from rFBApy.fba.fba_interface import FluxBalanceAnalysis
from rFBApy.rational_lp_solver import Fba


# ==============================================================================
# Flux Balance Analysis - Exact Rational Implementation
# ==============================================================================
class RationalFba(FluxBalanceAnalysis):
    # > Exact: values are not clipped (lpeps is ignored), and both steps of pFBA
    # > run in Rust, where the optimum fixed for the second step stays exact.

    def __init__(self: RationalFba, model: Fba, reactions: list[str]) -> None:
        super().__init__()
        self.model: Fba = model
        self.__reactions: list[str] = reactions
        self.__index: dict[str, int] = {r: i for i, r in enumerate(reactions)}

    # --------------------------------------------------------------------------
    # Setters
    # --------------------------------------------------------------------------
    # ~ Init
    @classmethod
    def _lpinit(cls: type[RationalFba], objective: str,
                stoichiometry: dict[str, set[tuple[float, str]]],
                bounds: dict[str, tuple[float, float]]) -> RationalFba:
        reactions: list[str] = list(bounds)
        index: dict[str, int] = {r: i for i, r in enumerate(reactions)}
        triples: list[tuple[int, int, float]] = [
            (row, index[r], coeff)
            for row, expr in enumerate(stoichiometry.values())
            for coeff, r in expr
        ]
        model: Fba = Fba(
            len(stoichiometry), triples, [bounds[r] for r in reactions],
            index[objective],
        )
        fba: RationalFba = RationalFba(model, reactions)
        fba._bounds = bounds.copy()
        return fba

    # --------------------------------------------------------------------------
    # Solving
    # --------------------------------------------------------------------------
    def solve(
        self: RationalFba,
        bounds: dict[str, tuple[float, float]] | None = None,
        pfba: bool = True,
    ) -> tuple[None | float, dict[str, float]]:
        overrides: list[tuple[int, float, float]] = [
            (self.__index[r], lb, ub) for r, (lb, ub) in (bounds or {}).items()
        ]
        opt, fluxes = self.model.solve(overrides, pfba)
        if opt is None:
            return None, {}
        return opt, dict(zip(self.__reactions, fluxes))
