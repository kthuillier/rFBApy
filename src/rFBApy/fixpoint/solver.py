# ==============================================================================
# Imports
# ==============================================================================
from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Iterator
from math import inf
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import clingo
from merrinasp.theory.language import THEORY_LANGUAGE, rewrite
from merrinasp.theory.propagator import LpPropagator

from rFBApy.asp import (
    GroundingContext,
    asp_number,
    asp_string,
    asp_value,
    instance_to_asp,
)
from rFBApy.fba import DEFAULT_SOLVER, LP_SOLVERS
from rFBApy.metabolic_network import MetabolicNetwork
from rFBApy.regulatory_network import RegulatoryNetwork
from rFBApy.rfba import FIXPOINT_ACCURACY, OPTIMALITY, state_bounds

from . import lpcore
from .cut import Cut

# ~ Type hints only: cache.py imports Solution from here
if TYPE_CHECKING:
    from .cache import Cache

# ==============================================================================
# Global
# ==============================================================================
ASP_DIR = Path(__file__).parent / "asp"
ENCODING = ASP_DIR / "fixpoint.lp"
OPTIMAL = ASP_DIR / "fixpoint.optimal.lp"

# ~ LP conflicts by bisection (lpcore) instead of merrinasp's own analysis
BISECTION = True
# ~ Largest number of cached cuts given to one solve
IMPORTED = 50

_FLUX_VARIABLE = re.compile(r'^f\("(.*)"\)$')


# ==============================================================================
# Solving
# ==============================================================================
class Solution(NamedTuple):
    ones: set[str]  # nodes at 1
    fluxes: dict[str, float]
    predicates: dict[tuple[str, str, float], bool]  # (variable, gt|lt, bound) -> value


def _predicate(term: clingo.Symbol) -> tuple[str, str, float]:
    variable, op, bound = term.arguments
    return (variable.string, op.name, float(asp_value(bound)))


def solve_instance(
    instance: str,
    lpsolver: str = DEFAULT_SOLVER,
    lpeps: float = FIXPOINT_ACCURACY,
    optimal: bool = False,
    n: int = 1,
    statistics: Counter[str] | None = None,
) -> Iterator[Solution]:
    # > One answer per regulatory state (#project). The encoding goes through
    # > merrinasp's rewriter; the instance, plain facts, is added as it is.
    if lpsolver not in LP_SOLVERS:
        raise ValueError(f"Unknown LP Solver: {lpsolver}.")
    options = [str(n), "--heuristic=Domain", "-c", f"eps={asp_number(lpeps)}"]
    if n != 1:
        options.append("--project")
    control = clingo.Control(options)
    if BISECTION:
        lpcore.install()
    else:
        lpcore.uninstall()
    propagator = LpPropagator(lpsolver=lpsolver)
    propagator.lazy(False)
    propagator.show_lpassignment(True)
    propagator.strict_forall_check(True)
    propagator.strict_epsilon(lpeps)
    control.register_propagator(propagator)  # type: ignore
    control.add("base", [], THEORY_LANGUAGE)
    rewrite(control, [str(file) for file in [ENCODING] + ([OPTIMAL] if optimal else [])])
    control.add("base", [], instance + "#show x/1.\n#show thr/3.\n")
    control.ground([("base", [])], context=GroundingContext())

    try:
        with control.solve(yield_=True) as handle:  # type: ignore
            for model in handle:
                # ~ thr/3: every predicate; x/1: nodes (strings) and predicates at 1
                ones: set[str] = set()
                predicates: dict[tuple[str, str, float], bool] = {}
                for atom in model.symbols(shown=True):
                    if atom.name == "thr":
                        predicates.setdefault(_predicate(atom), False)
                    elif atom.arguments[0].type is clingo.SymbolType.String:
                        ones.add(atom.arguments[0].string)
                    else:
                        predicates[_predicate(atom.arguments[0])] = True
                fluxes: dict[str, float] = {}
                for _, values in (propagator.get_assignment(model.thread_id) or {}).values():
                    for variable, value in (values or {}).items():
                        match = _FLUX_VARIABLE.match(variable)
                        if match is not None:
                            fluxes[match.group(1)] = float(value)  # pyright: ignore[reportArgumentType]
                yield Solution(ones, fluxes, predicates)
    finally:
        if statistics is not None:
            lp = propagator.get_statistics()
            solvers = control.statistics["solving"]["solvers"]
            statistics["asp solves"] += 1
            statistics["choices"] += int(solvers["choices"])
            statistics["conflicts"] += int(solvers["conflicts"])
            statistics["lp calls"] += int(lp["LP Solver"]["Solving"]["Calls"])  # type: ignore
            statistics["lp seconds"] += float(lp["LP Solver"]["Solving"]["Time (s)"])  # type: ignore
            statistics["lp conflicts"] += int(lp["Conflicts"]["Exists"])  # type: ignore


# ==============================================================================
# Optimal fixpoints
# ==============================================================================
def solve_optimal(
    mn: MetabolicNetwork,
    network: RegulatoryNetwork,
    obj: str,
    cache: Cache,
    medium: dict[str, float] = {},  # noqa: B006
    state: dict[str, int] = {},  # noqa: B006
    lpsolver: str = DEFAULT_SOLVER,
    lpeps: float = FIXPOINT_ACCURACY,
    opttol: float = OPTIMALITY,
    max_cuts: int = 100,
    n: int = 1,
    max_imported: int = IMPORTED,
    start: Solution | None = None,
) -> tuple[list[Solution], int]:
    # > Lazy optimality cuts. A candidate (x, v) is checked against the FBA
    # > optimum G of x. Below it, an optimal vector u of x gives the cut
    # > "wherever u is feasible, growth >= G" (fixpoint.optimal.lp), sound for
    # > every state and every instance of the network. pFBA's u only makes the
    # > cut apply to more states. Each round builds a new Control: merrinasp's
    # > propagator keeps only the constraints of the last grounding step, so
    # > answers already returned are passed back as found/2 facts.
    instance = instance_to_asp(mn, network, obj, medium, state)
    fba = cache.fba(mn, obj, lpsolver)
    regulated = mn.reactions().intersection(network.nodes)
    # ~ Cached cuts that fit the bounds of the instance; fixpoint.optimal.lp
    # ~ drops those whose reactions the state switches off
    cache.refresh()
    loaded = cache.select(obj, state_bounds(mn, medium), max_imported)
    cache.stats["cuts imported"] += len(loaded)
    hints = ""
    if start is not None:
        hints = "".join(
            f"hint({asp_string(node)},{1 if node in start.ones else -1}).\n"
            for node in sorted(network.nodes)
        ) + "".join(
            f"hint(thr({asp_string(variable)},{op},{asp_number(bound)}),{1 if holds else -1}).\n"
            for (variable, op, bound), holds in start.predicates.items()
        )
    elif loaded:
        hints = _hint_facts(loaded[0].fluxes, _flux_predicates(mn, network), lpeps)
    imported = len(loaded)
    found: list[Solution] = []
    excluded: list[str] = []
    fruitless = 0  # cuts since the last answer
    while True:
        cuts = "".join(_cut_facts(index, cut, opttol) for index, cut in enumerate(loaded, 1))
        candidates = solve_instance(
            instance + cuts + hints + "".join(excluded),
            lpsolver, lpeps, optimal=True, n=n - len(found) if n else 0,
            statistics=cache.stats,
        )
        try:
            for solution in candidates:
                bounds = state_bounds(mn, medium, regulated.difference(solution.ones))
                problem, optimum, vector = cache.maximum(fba, obj, bounds)
                if optimum is None or abs(optimum) == inf:
                    raise RuntimeError(
                        f"rFBApy finds no finite FBA optimum ({optimum!r}) for the "
                        "regulatory state of a fixpoint"
                    )
                margin = opttol * max(1.0, abs(optimum))
                if solution.fluxes[obj] < optimum - 2 * margin:
                    break  # ~ below the optimum: cut, new solver
                found.append(solution)
                fruitless = 0
                excluded.append(
                    f"% Fixpoint {len(found)}: already returned\nfound({len(found)}).\n"
                    + "".join(
                        f"found({len(found)},{asp_string(node)}).\n"
                        for node in sorted(solution.ones)
                    )
                )
            else:
                return found, len(loaded) - imported
        finally:
            candidates.close()  # pyright: ignore[reportAttributeAccessIssue]
        # ~ The cut of this state; u is the pFBA vector when it reaches the optimum
        cut = cache.cut_of(problem)
        if cut is None:
            if vector is None:
                _, vector = fba.solve(bounds, pfba=False)
            _, parsimonious = cache.fba(mn, obj, lpsolver, pfba=True).solve(bounds, pfba=True)
            if parsimonious.get(obj, -inf) >= optimum - margin:
                vector = parsimonious
            cut = Cut.of(obj, vector, optimum, problem, bounds)
        if any(cut.identity == other.identity for other in loaded):
            raise RuntimeError(
                f"the growth flux stays at {solution.fluxes[obj]!r}, below the rFBA optimum "
                f"{optimum!r} of its state, although the cut for that state is in the program: "
                "`opttol` is below what the LP solver can enforce, or the cut did not apply"
            )
        if fruitless >= max_cuts:
            raise RuntimeError(
                f"no optimal fixpoint found within {max_cuts} cuts"
                + (f", after the {len(found)} already found" if found else "")
            )
        fruitless += 1
        loaded.append(cut)
        cache.add(cut)
        reactions = mn.reactions()
        hints = _hint_facts(
            cut.fluxes, [p for p in solution.predicates if p[0] in reactions], lpeps
        )


def _cut_facts(index: int, cut: Cut, opttol: float) -> str:
    bound = cut.growth - opttol * max(1.0, abs(cut.growth))
    return (
        f"% Cut {index}: growth {cut.growth!r} with {len(cut.needs)} active reactions\n"
        + f"cut({index},{asp_number(bound)}).\n"
        + "".join(
            f"needs({index},{asp_string(reaction)},{direction}).\n"
            for reaction, direction in sorted(cut.needs)
        )
    )


def _hint_facts(
    fluxes: dict[str, float],
    predicates: Iterable[tuple[str, str, float]],
    lpeps: float,
) -> str:
    # ~ Value of each flux predicate in `fluxes` (missing reaction: flux 0)
    return "".join(
        f"hint(thr({asp_string(variable)},{op},{asp_number(bound)}),"
        f"{1 if (flux > bound + lpeps / 2 if op == 'gt' else flux < bound - lpeps / 2) else -1}).\n"
        for (variable, op, bound) in predicates
        for flux in [fluxes.get(variable, 0.0)]
    )


def _flux_predicates(mn: MetabolicNetwork, network: RegulatoryNetwork) -> set[tuple[str, str, float]]:
    # ~ Superset of the flux predicates of fixpoint.lp, known before solving:
    # ~ thresholds on reactions, and 0 on exchanges
    reactions = mn.reactions()
    exchanges = {mn.exchange(metabolite) for metabolite in mn.metabolites(True, False)}
    pairs = {(variable, float(value)) for variable, value in network.thresholds if variable in reactions}
    pairs |= {(exchange, 0.0) for exchange in exchanges if exchange is not None}
    return {(variable, op, bound) for variable, bound in pairs for op in ("gt", "lt")}
