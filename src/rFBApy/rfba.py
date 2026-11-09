# ==============================================================================
# Import
# ==============================================================================
from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from math import exp, floor, inf, isnan, log, log10
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd  # type: ignore

# ~ Custom modules
from rFBApy.asp import instance_to_asp
from rFBApy.bn.bexpr import (
    AndExpr,
    BoolExpr,
    NotExpr,
    OrExpr,
    ThresholdExpr,
    VarExpr,
)
from rFBApy.fba import ACCURACY, DEFAULT_SOLVER, FluxBalanceAnalysis
from rFBApy.metabolic_network import MetabolicNetwork
from rFBApy.regulatory_network import RegulatoryNetwork

# ~ Type hints only: the fixpoint package imports this module (see find_fixpoint)
if TYPE_CHECKING:
    from rFBApy.fixpoint.cache import Cache
    from rFBApy.fixpoint.solver import Solution

# ==============================================================================
# Global
# ==============================================================================
TAG_METABOLITE: str = "(met)"
TAG_REACTION: str = "(reaction)"
TAG_REGULATION: str = "(state)"
# ~ Columns added to the fixpoints by `verify`
COLUMN_VERIFIED: str = "verified"
COLUMN_OPTIMUM: str = "optimum"
COLUMN_DURATION: str = "duration"

BIOMASS: float = 1e-2
TAU: float = 1e-2
# ~ Fixpoints: smallest flux counted as non-zero (`eps` of fixpoint.lp), and
# ~ tolerance on the FBA optimum, relative to max(1, optimum)
FIXPOINT_ACCURACY: float = 1e-4
OPTIMALITY: float = 1e-6
# ~ find_fixpoint(cache=<path>): one Cache object per file and per process
_CACHES: dict[str, Cache] = {}

# ==============================================================================
# Auxiliary Functions
# ==============================================================================
# ------------------------------------------------------------------------------
# Input Functions
# ------------------------------------------------------------------------------
def load_networks(
    sbml: str | MetabolicNetwork,
    bnet: str | RegulatoryNetwork | None = None,
    sbmlqual: str | RegulatoryNetwork | None = None,
) -> tuple[MetabolicNetwork, RegulatoryNetwork | None]:
    if isinstance(sbml, MetabolicNetwork):
        mn: MetabolicNetwork = sbml
    else:
        mn: MetabolicNetwork = MetabolicNetwork.read_sbml(sbml)

    if isinstance(bnet, RegulatoryNetwork):
        bn: RegulatoryNetwork | None = bnet
    elif bnet is not None:
        bn: RegulatoryNetwork | None = RegulatoryNetwork.load_bnet(bnet)
    elif isinstance(sbmlqual, RegulatoryNetwork):
        bn: RegulatoryNetwork | None = sbmlqual
    elif sbmlqual is not None:
        bn: RegulatoryNetwork | None = RegulatoryNetwork.load_sbml(sbmlqual)
    else:
        bn: RegulatoryNetwork | None = None
    return mn, bn


def init_regulatory_network(
    mn: MetabolicNetwork,
    bn: RegulatoryNetwork | None,
    mutations: dict[str, int] = {},  # noqa: B006
    settings: dict[str, bool | int | float] = {},  # noqa: B006
    verbose: bool = True,
) -> RegulatoryNetwork | None:
    # ~ Completes `bn` in place with what the simulation evaluates
    if (len(mutations) != 0 or len(mn.genes_association()) != 0) and bn is None:
        bn = RegulatoryNetwork()
    if bn is None:
        return None

    # ~ Add gene association rules and missing genes to BN
    for gene, rule in mn.genes_association().items():
        if rule != "":
            bn[gene] = rule
    for gene in mn.genes():
        if gene not in bn:
            bn[gene] = 1

    for m in mn.metabolites(True, False):
        bn[m] = 0

    # ~ Special case: missing regulatory rule constant
    undefined_gene: frozenset[str] = bn.undefined.difference(
        mn.metabolites(True, False)
        .union(mn.reactions())
        .union(settings.keys())
    )
    thresholds: set[str] = {n for n, _ in bn.thresholds}
    for gene in undefined_gene:
        if gene in thresholds:
            bn[gene] = 0
        else:
            if verbose:
                print(f"Warning: {gene} is undefined. Set to 1.")
            bn[gene] = 1

    # ~ Experiment mutations
    for n, val in mutations.items():
        bn[n] = val
    return bn


# ------------------------------------------------------------------------------
# Update Functions
# ------------------------------------------------------------------------------
def update_regulatory_state(
    mn: MetabolicNetwork,
    bn: RegulatoryNetwork,
    w: dict[str, float],
    v: dict[str, float],
    x: dict[str, int | None],
    settings: dict[str, int | bool | float] = {},  # noqa: B006
) -> dict[str, int | None]:
    mext_state = {
        m: w.get(m, 0)
        for m in mn.metabolites(True, False)
    }
    r_state = {
        r: v.get(r, 0)
        for r in mn.reactions()
    }
    next_bn = bn(x | mext_state | r_state | settings)  # type: ignore
    return next_bn


def update_biomass(
    tau: float,
    biomass: float,
    growth: float,
    k: int = 1,
) -> float:
    return biomass * exp(growth * tau * k)


def update_kinetics(
    mn: MetabolicNetwork,
    tau: float,
    biomass: float,
    growth: float,
    w: dict[str, float],
    v: dict[str, float],
    k: int = 1,
):
    w_: dict[str, float] = w.copy()
    if growth == 0:
        return w_
    for n, wn_ in w_.items():
        r: str | None = mn.exchange(n)
        assert r is not None
        s_mr: float = mn.stoichiometry()[(n, r)]
        w_[n] = wn_ - ((s_mr * v[r]) / growth) * biomass * (
            1 - exp(growth * tau * k)
        )
        w_[n] = max(0.0, w_[n])
    return w_


def update_bounds(
    mn: MetabolicNetwork,
    tau: float,
    biomass: float,
    w: dict[str, float],
) -> dict[str, tuple[float, float]]:
    updated_bounds: dict[str, tuple[float, float]] = {}
    for m in mn.metabolites(external=True, internal=False):
        r: str | None = mn.exchange(m)
        assert r is not None
        s_tr: float = mn.stoichiometry()[(m, r)]
        lb, ub = mn.bound(r)
        max_rate: float = max(w[m] / (tau * biomass), 0)

        if s_tr < 0:
            low_bound: float = lb
            up_bound: float = min(ub, max_rate)
        else:
            low_bound: float = max(lb, -max_rate)
            up_bound: float = ub

        updated_bounds[r] = (low_bound, up_bound)
    return updated_bounds


# ------------------------------------------------------------------------------
# Time Step Controller
# ------------------------------------------------------------------------------
def estimate_metabolite_threshold_crossing_ts(
    mn: MetabolicNetwork,
    obj: str,
    m: str,
    tau: float,
    biomass: float,
    b: float,
    w: dict[str, float],
    v: dict[str, float],
) -> int:
    if w[m] == b:
        return inf  # type: ignore  # déjà sur la borne

    r: str | None = mn.exchange(m)
    assert r is not None
    s_mr: float = mn.stoichiometry()[(m, r)]

    if (v[r] == 0) or (v[obj] == 0):
        return inf  # type: ignore  # flux ou croissance nul -> pas de croisement

    direction = s_mr * v[r]  # signe de dW/dt (à un facteur biomass>0 près)

    if w[m] < b and direction <= 0:
        return inf  # type: ignore  # ne montera jamais au-dessus de b
    if w[m] > b and direction >= 0:
        return inf  # type: ignore  # ne descendra jamais sous b

    arg = 1 + ((b - w[m]) * v[obj]) / (s_mr * v[r] * biomass)
    if arg <= 0:
        return inf  # type: ignore  # asymptote : borne jamais atteinte

    return floor(log(arg) / (v[obj] * tau))


def estimate_mn_state_duration_ts(
    mn: MetabolicNetwork,
    thresholds: set[tuple[str, float]],
    obj: str,
    biomass: float,
    tau: float,
    w: dict[str, float],
    v: dict[str, float],
) -> int:
    metabolites: set[str] = mn.metabolites(True, False)
    thresholds = thresholds.union((m, 0.0) for m in metabolites)
    return min(
        estimate_metabolite_threshold_crossing_ts(mn, obj, m, tau, biomass, b, w, v)
        for m, b in thresholds
        if m in metabolites
    )


def estimate_state_duration(
    mn: MetabolicNetwork,
    bn: RegulatoryNetwork | None,
    tau: float,
    biomass: float,
    x: dict[str, int | None],
    v: dict[str, float],
    w: dict[str, float],
    obj: str,
    settings: dict[str, int | bool | float] = {},  # noqa: B006
) -> int:
    biomass_plus_1: float = update_biomass(tau, biomass, v[obj], k=1)
    w_plus_1: dict[str, float] = update_kinetics(
        mn,
        tau,
        biomass_plus_1,
        v[obj],
        w,
        v,
        k=1,
    )
    x_plus_1: dict[str, int | None] = {}
    if bn is not None:
        x_plus_1 = update_regulatory_state(
            mn,
            bn,
            w_plus_1,
            v,
            x,
            settings=settings,
        )
    if x != x_plus_1:
        return 1

    bounds: set[tuple[str, float]] = set() if bn is None else bn.thresholds  # type: ignore
    return max(
        1, estimate_mn_state_duration_ts(mn, bounds, obj, biomass, tau, w, v)
    )


# ==============================================================================
# Main Functions
# ==============================================================================
def next_iter(
    mn: MetabolicNetwork,
    bn: RegulatoryNetwork | None,
    fba: FluxBalanceAnalysis,
    obj: str,
    tau: float,
    v: dict[str, float],
    w: dict[str, float],
    x: dict[str, int | None],
    biomass: float,
    duration: int,
    settings: dict[str, bool | int | float] = {},  # noqa: B006
    pfba: bool = True,
) -> tuple[dict[str, float], dict[str, float], dict[str, int | None], float, int]:
    next_biomass: float = update_biomass(tau, biomass, v[obj], duration)

    next_w: dict[str, float] = update_kinetics(
        mn,
        tau,
        biomass,
        v[obj],
        w,
        v,
        duration,
    )
    next_x: dict[str, int | None] = {}
    bounds: dict[str, tuple[float, float]] = update_bounds(
        mn,
        tau,
        next_biomass,
        next_w,
    )
    if bn is not None:
        next_x = update_regulatory_state(mn, bn, next_w, v, x, settings)
        bounds |= {
            n: (0.0, 0.0)
            for n in bn
            if n in mn.reactions() and next_x[n] is not None and next_x[n] == 0
        }
    
    opt, next_v = fba.solve(bounds, pfba=pfba)
    if opt is None or opt < 0:  # FIXME
        opt = 0.0
        next_v = {r: 0.0 for r in mn.reactions()}

    next_duration: int = estimate_state_duration(
        mn,
        bn,
        tau,
        next_biomass,
        next_x,
        next_v,
        next_w,
        obj,
        settings=settings,
    )

    return (next_v, next_w, next_x, next_biomass, next_duration)


def simulate_rfba(
    sbml: str | MetabolicNetwork,
    obj: str,
    bnet: str | RegulatoryNetwork | None = None,
    sbmlqual: str | RegulatoryNetwork | None = None,
    concentrations: dict[str, float] = {},  # noqa: B006
    state: dict[str, int] = {},  # noqa: B006
    bounds: dict[str, tuple[float, float]] = {},  # noqa: B006
    mutations: dict[str, int] = {},  # noqa: B006
    settings: dict[str, bool | int | float] = {},  # noqa: B006
    biomass: float = BIOMASS,
    tau: float = TAU,
    iter: int = 100,
    compressed: bool = False,
    lpsolver: str = DEFAULT_SOLVER,
    lpeps: float = ACCURACY,
    pfba: bool = True,
) -> pd.DataFrame:
    # --------------------------------------------------------------------------
    # Pre-processing
    # --------------------------------------------------------------------------
    # ~ Inputs
    mn, bn = load_networks(sbml, bnet, sbmlqual)

    # ~ Experiment bounds
    #
    # Must run BEFORE instantiate_fba(): MetabolicNetwork.instantiate_fba()
    # passes self.__bounds.copy() to the LP backend as a one-time snapshot
    # (FluxBalanceAnalysis._bounds), not a live view of `mn`. Every simulation
    # tick's solve() call only overrides that snapshot for the reactions
    # explicitly present in its own `bounds` dict argument -- exchange
    # reactions (recomputed each tick by update_bounds() from mn's *current*
    # bounds) and reactions a regulatory gene switches off. A non-exchange
    # (internal) reaction customized only via this function's `bounds`
    # argument is never in either of those per-tick sets, so if set_bound()
    # ran after instantiate_fba(), the override would silently never reach
    # the solver -- the reaction would keep using whatever bound the raw
    # SBML declared, for the entire simulation.
    for r, (lb, ub) in bounds.items():
        if r in mn.reactions():
            mn.set_bound(r, lb, ub)

    fba: FluxBalanceAnalysis = mn.instantiate_fba(obj, lpsolver=lpsolver, lpeps=lpeps)

    bn = init_regulatory_network(mn, bn, mutations, settings)

    # --------------------------------------------------------------------------
    # Simulation
    # --------------------------------------------------------------------------
    v: dict[str, float] = {r: 0.0 for r in mn.reactions()}
    w: dict[str, float] = {
        m: concentrations.get(m, 0.0) for m in mn.metabolites(True, False)
    }
    if bn is not None:
        x: dict[str, int | None] = bn(
            {n: None for n in bn if n not in mn.metabolites(True, False)}
            | {n: 1 if w.get(n, 0) > 0 else 0 for n in mn.metabolites(True, False)}
            | {n: None for n in mn.reactions()}
            | state  # type: ignore
            | mutations
            | settings
        )
        x |= {n: None for n in bn if n in mn.reactions()}
    else:
        x: dict[str, int | None] = {}

    simulation_states: list[
        tuple[
            dict[str, float],
            dict[str, float],
            dict[str, int | None],
            float,
            int,
        ]
    ] = [(v, w, x, biomass, 1)]
    iter += 1
    i = 1
    while i < iter + 1:
        v_0, w_0, x_0, biomass_0, duration_0 = simulation_states[-1]
        v_1, w_1, x_1, biomass_1, duration_1 = next_iter(
            mn,
            bn,
            fba,
            obj,
            tau,
            v_0,
            w_0,
            x_0,
            biomass_0,
            duration_0,
            settings=settings,
            pfba=pfba,
        )
        if i + duration_1 >= iter + 1:
            diff_i_iter = iter + 1 - i
            duration_1 = diff_i_iter
        simulation_states.append((v_1, w_1, x_1, biomass_1, duration_1))
        i += duration_1

    # --------------------------------------------------------------------------
    # Post-processing
    # --------------------------------------------------------------------------
    met_cols: dict[str, str] = {
        m: f"{m} {TAG_METABOLITE}" for m in mn.metabolites(True, False)
    }
    react_cols: dict[str, str] = {r: f"{r} {TAG_REACTION}" for r in mn.reactions()}
    reg_cols: dict[str, str] = {}
    if bn is not None:
        reg_cols |= {
            n: f"{n} {TAG_REGULATION}" for n in bn if n not in mn.metabolites()
        }

    sim: list[dict[str, float | int | None]] = []
    iter_: int = 0
    for s in range(len(simulation_states)):
        v_0, w_0, x_0, biomass_0, duration_0 = simulation_states[s]
        growth_0 = v_0[obj]
        # ~ Fluxes and regulatory states are constant over the interval
        constant_0: dict[str, float | int | None] = {
            react_cols[r]: v_0[r] for r in v_0
        }
        if bn is not None:
            constant_0 |= {reg_cols[n]: x_0[n] for n in x_0 if n in reg_cols}
        for i in range(duration_0) if not compressed else {0, duration_0 - 1}:
            time: float = (iter_ + i) * tau
            time_rounded: float = round(time, abs(floor(log10(tau))))
            biomass_i: float = update_biomass(tau, biomass_0, growth_0, i + 1)
            w_i: dict[str, float] = update_kinetics(
                mn, tau, biomass_0, growth_0, w_0, v_0, i + 1
            )
            state_i: dict[str, float | int | None] = (
                {
                    "Time": time_rounded,
                    "biomass": biomass_i,
                }
                | {met_cols[m]: w_i[m] for m in w_i}
                | constant_0
            )  # type: ignore
            sim.append(state_i)

        iter_ += duration_0

    cols: list[str] = [
        "Time",
        *sorted(met_cols.values()),
        *sorted(react_cols.values()),
        *(sorted(reg_cols.values()) if bn is not None else []),
        "biomass",
    ]
    sim_df: pd.DataFrame = pd.DataFrame(sim, columns=cols)

    return sim_df


# ==============================================================================
# Fixpoint Verification
# ==============================================================================
def state_bounds(
    mn: MetabolicNetwork,
    medium: dict[str, float],
    off: Iterable[str] = (),
) -> dict[str, tuple[float, float]]:
    # ~ All the bounds, restricted as in fixpoint.lp: no uptake of an external
    # ~ metabolite the medium does not feed, no flux through a reaction of `off`
    bounds = dict(mn.bounds())
    stoichiometry = mn.stoichiometry()
    for metabolite in mn.metabolites(True, False):
        exchange = mn.exchange(metabolite)
        if exchange is None or float(medium.get(metabolite, 0)) > 0:
            continue
        lower, upper = bounds[exchange]
        if stoichiometry[(metabolite, exchange)] < 0:
            bounds[exchange] = (lower, min(upper, 0.0))
        else:
            bounds[exchange] = (max(lower, 0.0), upper)
    for reaction in off:
        bounds[reaction] = (0.0, 0.0)
    return bounds


def check_fixpoint(
    mn: MetabolicNetwork,
    bn: RegulatoryNetwork | None,
    obj: str,
    state: dict[str, int],
    fluxes: dict[str, float],
    concentrations: dict[str, float],
    mutations: dict[str, int] = {},  # noqa: B006
    settings: dict[str, bool | int | float] = {},  # noqa: B006
    lpeps: float = FIXPOINT_ACCURACY,
    optimal: bool = False,
    opttol: float = OPTIMALITY,
    lpsolver: str = DEFAULT_SOLVER,
    secreted: frozenset[str] | set[str] = frozenset(),
    biomass: float = BIOMASS,
    tau: float = TAU,
) -> tuple[list[str], float | None, float]:
    # > Without ASP: the network is built by the code of simulate_rfba, the
    # > state is updated once and must not change, the flux vector is tested
    # > directly, and the FBA optimum and the duration come from rFBApy. A flux
    # > below lpeps / 10 reads as 0 for the regulation only. Returns the failed
    # > conditions (none: it is a fixpoint), the FBA optimum of the state, and
    # > how long it holds (in the time unit of tau, inf if no threshold is crossed).
    tolerance = lpeps / 10
    external = mn.metabolites(True, False)
    reactions = mn.reactions()
    stoichiometry = mn.stoichiometry()
    v = {r: float(fluxes.get(r, 0.0)) for r in reactions}
    v_read = {r: 0.0 if abs(flux) < tolerance else flux for r, flux in v.items()}
    w = {m: float(concentrations.get(m, 0.0)) for m in external}
    provided = {m for m in external if w[m] > 0}.difference(secreted)
    x: dict[str, int | None] = {node: int(value) for node, value in state.items()}
    problems: list[str] = []

    def report(what: str, items: list[str]) -> None:
        if items:
            shown = ", ".join(items[:5]) + (", ..." if len(items) > 5 else "")
            problems.append(f"{what}: {len(items)} ({shown})")

    # ~ The regulatory state must not change
    reference = init_regulatory_network(
        mn, None if bn is None else bn.copy(), mutations, settings, verbose=False
    )
    updated: dict[str, int | None] = {}
    switched_off: set[str] = set()
    if reference is not None:
        updated = update_regulatory_state(mn, reference, w, v_read, x, dict(settings))  # type: ignore
        nodes = [node for node in reference if node not in external]
        report(
            "nodes changed by the regulatory update of rFBApy",
            [f"{n}: {x.get(n)} -> {updated[n]}" for n in nodes if updated[n] != x.get(n)],
        )
        switched_off = {n for n in nodes if n in reactions and updated[n] == 0}
    changed = bool(problems)

    # ~ The flux vector must fit the updated state
    report("reactions switched off by the regulation that carry flux", [
        f"{r}: {v[r]:g}" for r in sorted(switched_off) if abs(v[r]) > tolerance
    ])
    report("fluxes out of bounds", [
        f"{r}: {v[r]:g} not in [{lower:g}, {upper:g}]"
        for r, (lower, upper) in sorted(mn.bounds().items())
        if not lower - tolerance <= v[r] <= upper + tolerance
    ])
    balance: dict[str, float] = {}
    largest: dict[str, float] = {}
    for (metabolite, reaction), coefficient in stoichiometry.items():
        if metabolite not in external:
            term = coefficient * v[reaction]
            balance[metabolite] = balance.get(metabolite, 0.0) + term
            largest[metabolite] = max(largest.get(metabolite, 1.0), abs(term))
    report("internal metabolites not at steady state", [
        f"{m}: {b:g}" for m, b in sorted(balance.items()) if abs(b) > tolerance * largest[m]
    ])
    report("metabolites taken up that the medium does not provide", [
        f"{m} by {mn.exchange(m)}"
        for m in sorted(external)
        if mn.exchange(m) is not None
        and m not in provided
        and stoichiometry[(m, mn.exchange(m))] * v[mn.exchange(m)] < -tolerance  # type: ignore
    ])
    if v.get(obj, 0.0) < lpeps - tolerance:
        problems.append(f"flux of the objective {obj} below {lpeps:g}: {v.get(obj, 0.0):g}")

    # ~ The FBA optimum of the updated state; flux predicates are read-outs, not bounds
    # ~ lpeps=None: the fixpoint reads raw fluxes with its own thresholds
    fba = mn.instantiate_fba(obj, lpsolver, lpeps=None)
    medium = {m: w[m] for m in provided}
    optimum, _ = fba.solve(state_bounds(mn, medium, switched_off), pfba=False)
    if optimum is None:
        problems.append("rFBApy finds no FBA optimum for this regulatory state")
    elif optimal and v.get(obj, 0.0) < optimum - 2 * opttol * max(1.0, abs(optimum)) - tolerance / 10:
        problems.append(
            f"flux of the objective {obj} below the rFBA optimum of this state: "
            f"{v.get(obj, 0.0):.9g} against {optimum:.9g}"
        )

    # ~ How long the state holds, from the fluxes of the fixpoint
    duration = 0.0
    if not changed:
        steps = estimate_state_duration(
            mn, reference, tau, biomass, updated, v_read, w, obj, dict(settings)  # type: ignore
        )
        duration = steps * tau
    return problems, optimum, duration


# ==============================================================================
# Fixpoint Search
# ==============================================================================
def _read_as_thresholds(expr: BoolExpr, operators: dict[str, str]) -> BoolExpr:
    # ~ A bare variable v of `operators` becomes [v operators[v] 0]
    if isinstance(expr, (AndExpr, OrExpr)):
        return type(expr)([_read_as_thresholds(item, operators) for item in expr.items])
    if isinstance(expr, NotExpr):
        return NotExpr(_read_as_thresholds(expr.item, operators))
    if isinstance(expr, VarExpr) and expr.name in operators:
        return ThresholdExpr(VarExpr(expr.name), operators[expr.name], 0.0)
    return expr


def bare_variables(expr: BoolExpr) -> set[str]:
    # ~ Variables read as Booleans, outside any threshold
    if isinstance(expr, (AndExpr, OrExpr)):
        return set().union(*(bare_variables(item) for item in expr.items))
    if isinstance(expr, NotExpr):
        return bare_variables(expr.item)
    return {expr.name} if isinstance(expr, VarExpr) else set()


def regulated_network(
    mn: MetabolicNetwork,
    bn: RegulatoryNetwork | None = None,
    mutations: dict[str, int] = {},  # noqa: B006
    settings: dict[str, bool | int | float] = {},  # noqa: B006
) -> RegulatoryNetwork:
    # > The network simulate_rfba evaluates, as rules: gene associations, genes
    # > without rule at 1, mutations as constants, Boolean settings as constants,
    # > and a bare metabolite M (reaction R) read as [M > 0] ([R != 0]).
    # > Undefined terms keep the defaults of simulate_rfba (0 in a threshold, else 1).
    network = RegulatoryNetwork() if bn is None else bn.copy()
    for reaction, rule in mn.genes_association().items():
        if rule != "":
            network[reaction] = rule
    for gene in mn.genes().difference(network.nodes):
        network[gene] = 1

    operators = {metabolite: ">" for metabolite in mn.metabolites(True, False)}
    operators |= {reaction: "!=" for reaction in mn.reactions()}
    for node in list(network):
        network[node] = _read_as_thresholds(network[node], operators)

    for node, value in mutations.items():
        network[node] = int(value)

    nodes = network.nodes
    read = set().union(*(bare_variables(network[node]) for node in nodes))
    for name, value in settings.items():
        if name in nodes:
            raise ValueError(f"setting '{name}' is a node with a rule: use `mutations` to force it")
        if name in read:
            network[name] = 1 if value else 0

    # ~ Also reports undefined terms set to 0, unlike simulate_rfba
    thresholds = {variable for variable, _ in network.thresholds}
    defined = network.nodes.union(mn.metabolites(True, False), mn.reactions(), settings)
    for name in sorted((read | thresholds).difference(defined)):
        print(f"Warning: {name} is undefined. Set to {0 if name in thresholds else 1}.")
    return network


def _entry_level(predicates: list[tuple[str, float, bool]]) -> float:
    # ~ A secreted metabolite enters its interval of predicates at the low end
    # ~ (just above it when open): that is where the duration is counted from
    lower, strict, upper = 0.0, True, inf
    for op, bound, holds in predicates:
        if (op == "gt") == holds:
            if bound > lower or (bound == lower and op == "gt"):
                lower, strict = bound, op == "gt"
        else:
            upper = min(upper, bound)
    if not strict or upper <= lower:
        return lower
    return lower + min(1e-6 * max(1.0, abs(lower)), (upper - lower) / 2)


def find_fixpoint(
    sbml: str | MetabolicNetwork,
    obj: str,
    bnet: str | RegulatoryNetwork | None = None,
    sbmlqual: str | RegulatoryNetwork | None = None,
    concentrations: dict[str, float] = {},  # noqa: B006
    state: dict[str, int] = {},  # noqa: B006
    bounds: dict[str, tuple[float, float]] = {},  # noqa: B006
    mutations: dict[str, int] = {},  # noqa: B006
    settings: dict[str, bool | int | float] = {},  # noqa: B006
    lpsolver: str = DEFAULT_SOLVER,
    lpeps: float = FIXPOINT_ACCURACY,
    verify: bool = False,
    optimal: bool = False,
    opttol: float = OPTIMALITY,
    n: int = 1,
    cache: Cache | str | Path | None = None,
    biomass: float = BIOMASS,
    tau: float = TAU,
) -> pd.DataFrame | None:
    # > A fixpoint is a regulatory state x unchanged by one update (rules read x,
    # > the medium and the fluxes v), with v at steady state, within bounds,
    # > taking up only the medium, zero on switched-off reactions, and obj >= lpeps.
    # > `optimal`: v also reaches the FBA optimum of x. `n`: up to n regulatory
    # > states, 0 for all. One row per fixpoint, None if there is none.
    # ~ Imported here: the fixpoint package imports this module, and only a
    # ~ fixpoint search needs merrinasp
    from rFBApy.fixpoint.cache import Cache
    from rFBApy.fixpoint.solver import solve_instance, solve_optimal

    if n < 0:
        raise ValueError(f"n is the largest number of fixpoints, 0 for all: got {n}")
    if isinstance(cache, (str, Path)):
        file = str(Path(cache).resolve())
        if file not in _CACHES:
            _CACHES[file] = Cache(file)
        cache = _CACHES[file]

    # --------------------------------------------------------------------------
    # Inputs
    # --------------------------------------------------------------------------
    mn, bn = load_networks(sbml, bnet, sbmlqual)
    mn = mn.copy()
    if cache is not None:
        cache.bind(mn)
    for reaction, (lower, upper) in bounds.items():
        if reaction in mn.reactions():
            mn.set_bound(reaction, lower, upper)
    network = regulated_network(mn, bn, mutations, settings)
    # ~ A numeric setting (e.g. "pH") is read through thresholds, like a concentration
    medium = dict(concentrations) | {name: float(value) for name, value in settings.items()}

    # --------------------------------------------------------------------------
    # Solving
    # --------------------------------------------------------------------------
    cuts = 0
    solutions: list[Solution] | None = None
    start: Solution | None = None
    instance = ""
    if cache is not None:
        cache.refresh()
        # ~ No growth with every reaction available: no fixpoint, whatever the strain
        unregulated = state_bounds(mn, medium)
        problem, ceiling, _ = cache.maximum(cache.fba(mn, obj, lpsolver), obj, unregulated)
        if ceiling is None or ceiling < lpeps / 2:
            cache.stats["no growth without regulation"] += 1
            solutions = []
        elif n == 1:
            # ~ Key of the instance without its mutations: a stored fixpoint is
            # ~ reused if the mutations keep it, else it seeds the search
            instance = hashlib.sha1(json.dumps([
                cache.regulatory_fingerprint(bn), problem,
                sorted((name, float(value)) for name, value in medium.items() if float(value) != 0),
                sorted((name, repr(value)) for name, value in settings.items()),
                sorted((node, int(value)) for node, value in state.items()),
                lpeps, optimal, opttol if optimal else None,
            ]).encode()).hexdigest()
            stored = cache.fixpoint(instance, mutations, mn.reactions())
            if stored is not None:
                solutions = [stored]
            else:
                start = cache.neighbour(instance)
    if solutions is None:
        if optimal:
            solutions, cuts = solve_optimal(
                mn, network, obj, Cache() if cache is None else cache, medium, state,
                lpsolver, lpeps, opttol, n=n, start=start,
            )
        else:
            solutions = list(solve_instance(
                instance_to_asp(mn, network, obj, medium, state), lpsolver, lpeps, n=n,
                statistics=cache.stats if cache is not None else None,
            ))
        if cache is not None and instance and solutions and not mutations:
            cache.add_fixpoint(instance, mutations, solutions[0])
    if not solutions:
        if verify:
            print("Fixpoint check with rFBApy: nothing to check, no fixpoint was found")
        return None

    # --------------------------------------------------------------------------
    # Output
    # --------------------------------------------------------------------------
    # ~ Nodes with a rule, and those only read by rules (default values), as simulate_rfba
    nodes = network.nodes
    nodes |= set().union(*(bare_variables(network[node]) for node in nodes))
    nodes |= {variable for variable, _ in network.thresholds}.difference(
        mn.metabolites(True, False), mn.reactions(), medium
    )
    stoichiometry = mn.stoichiometry()

    rows: list[dict[str, float | int | bool]] = []
    for index, solution in enumerate(solutions, start=1):
        fluxes = solution.fluxes
        # ~ `levels` is shown (NaN: secreted, positive with no value); `witness`
        # ~ gives a secreted metabolite a value that fits its predicates, for the check
        levels: dict[str, float] = {}
        witness: dict[str, float] = {}
        for metabolite in sorted(mn.metabolites(True, False)):
            exchange = mn.exchange(metabolite)
            if float(medium.get(metabolite, 0)) != 0:
                levels[metabolite] = witness[metabolite] = float(medium[metabolite])
            elif exchange is not None and (
                stoichiometry[(metabolite, exchange)] * fluxes.get(exchange, 0.0) > lpeps / 2
            ):
                levels[metabolite] = float("nan")
                witness[metabolite] = _entry_level([
                    (op, bound, holds)
                    for (variable, op, bound), holds in solution.predicates.items()
                    if variable == metabolite
                ])
            else:
                levels[metabolite] = witness[metabolite] = 0.0
        values = {node: int(node in solution.ones) for node in sorted(nodes)}

        row: dict[str, float | int | bool] = {}
        for metabolite, level in levels.items():
            row[f"{metabolite} {TAG_METABOLITE}"] = level
        for reaction in sorted(mn.reactions()):
            row[f"{reaction} {TAG_REACTION}"] = fluxes.get(reaction, 0.0)
        for node, value in values.items():
            row[f"{node} {TAG_REGULATION}"] = value
        if verify:
            problems, optimum, duration = check_fixpoint(
                mn, bn, obj, values, fluxes, witness, mutations, settings, lpeps,
                optimal, opttol, lpsolver,
                secreted={metabolite for metabolite, level in levels.items() if isnan(level)},
                biomass=biomass, tau=tau,
            )
            row[COLUMN_VERIFIED] = not problems
            row[COLUMN_OPTIMUM] = float("nan") if optimum is None else optimum
            row[COLUMN_DURATION] = duration
            what = "optimal fixpoint" if optimal else "fixpoint"
            if len(solutions) > 1:
                what += f" {index} of {len(solutions)}"
            if optimal:
                what += f", {cuts} cut(s)"
            shown = "none" if optimum is None else f"{optimum:.6g}"
            print(
                f"Fixpoint check with rFBApy ({what}): " + ("FAILED" if problems else "OK")
                + f" | rFBA optimum {shown} | holds {duration:g}"
            )
            for problem in problems:
                print(f"  - {problem}")
        rows.append(row)
    return pd.DataFrame(rows)


# ==============================================================================
# Main
# ==============================================================================
if __name__ == "__main__":
    ...
