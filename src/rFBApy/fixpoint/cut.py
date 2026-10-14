# ==============================================================================
# Imports
# ==============================================================================
from __future__ import annotations

from dataclasses import dataclass, field

# ==============================================================================
# Global
# ==============================================================================
# ~ A smaller flux counts as "reaction not used"
UNUSED = 1e-9
# ~ How far a cut flux may exceed a bound of the instance reusing it
_FIT = 1e-9


# ==============================================================================
# Cut
# ==============================================================================
# > An optimal flux vector u of some regulatory state, with its growth G:
# > wherever u is feasible, the optimum is at least G (fixpoint.optimal.lp).
# ==============================================================================
@dataclass
class Cut:
    objective: str
    growth: float
    fluxes: dict[str, float]  # only the reactions that carry flux
    problem: str = ""  # key of the FBA problem it solves
    needs: frozenset[tuple[str, int]] = field(init=False, repr=False, compare=False)

    def __post_init__(self: Cut) -> None:
        self.needs = frozenset((r, 1 if flux > 0 else -1) for r, flux in self.fluxes.items())

    @classmethod
    def of(
        cls: type[Cut],
        objective: str,
        vector: dict[str, float],
        optimum: float,
        problem: str = "",
        bounds: dict[str, tuple[float, float]] | None = None,
    ) -> Cut:
        # ~ Fluxes clipped to the bounds they were solved under (GLPK may return
        # ~ -1e-9 on a lower bound of 0), growth no higher than the vector carries
        if bounds is not None:
            vector = {
                r: min(max(flux, bounds[r][0]), bounds[r][1]) if r in bounds else flux
                for r, flux in vector.items()
            }
        fluxes = {r: float(flux) for r, flux in vector.items() if abs(flux) > UNUSED}
        return cls(objective, min(float(optimum), float(vector[objective])), fluxes, problem)

    @property
    def identity(self: Cut) -> tuple[str, float, frozenset[tuple[str, int]]]:
        return (self.objective, self.growth, self.needs)

    def fits(self: Cut, bounds: dict[str, tuple[float, float]]) -> bool:
        for reaction, flux in self.fluxes.items():
            if reaction not in bounds:
                return False
            lower, upper = bounds[reaction]
            below = flux < lower - _FIT * max(1.0, abs(lower))
            if below or flux > upper + _FIT * max(1.0, abs(upper)):
                return False
        return True
