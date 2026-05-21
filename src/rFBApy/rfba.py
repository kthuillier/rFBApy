# ==============================================================================
# Import
# ==============================================================================
from __future__ import annotations

from math import exp, floor, inf, log

import pandas as pd  # type: ignore
from bonesis import BooleanNetwork  # type: ignore

from rFBApy.fba import FluxBalanceAnalysis, DEFAULT_SOLVER

# ~ Custom modules
from rFBApy.metabolic_network import MetabolicNetwork


# ==============================================================================
# Global
# ==============================================================================
TAG_METABOLITE: str = "(met)"
TAG_REACTION: str = "(reaction)"
TAG_REGULATION: str = "(state)"


# ==============================================================================
# Auxiliary Functions
# ==============================================================================
# ------------------------------------------------------------------------------
# Update Functions
# ------------------------------------------------------------------------------
def update_regulatory_state(
    mn: MetabolicNetwork,
    bn: BooleanNetwork,
    w: dict[str, float],
    v: dict[str, float],
    x: dict[str, int],
) -> dict[str, int]:
    mext_state = {m: 0 if w.get(m, 0) == 0 else 1 for m in mn.metabolites(True, False)}
    r_state = {r: 0 for r in mn.reactions() if v.get(r, 0) == 0 and r in bn}
    next_bn = bn(x | mext_state | r_state) | mext_state
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
    for n in w_:
        r: str | None = mn.exchange(n)
        assert r is not None
        s_mr: float = mn.stoichiometry()[(n, r)]
        w_[n] = w_[n] = w_[n] - ((s_mr * v[r]) / growth) * biomass * (
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
    for t in mn.metabolites(external=True, internal=False):
        r: str | None = mn.exchange(t)
        assert r is not None
        s_tr: float = mn.stoichiometry()[(t, r)]
        if s_tr == 1:
            continue
        lb, ub = mn.bound(r)
        low_bound: float = lb
        up_bound: float = min(ub, max((w[t] / (tau * biomass)), 0))
        updated_bounds[r] = (low_bound, up_bound)
    return updated_bounds


# ------------------------------------------------------------------------------
# Time Step Controller
# ------------------------------------------------------------------------------
def estimate_metabolite_depletion_ts(
    mn: MetabolicNetwork,
    obj: str,
    m: str,
    tau: float,
    biomass: float,
    w: dict[str, float],
    v: dict[str, float],
) -> int:
    r: str | None = mn.exchange(m)
    assert r is not None
    s_mr: float = mn.stoichiometry()[(m, r)]
    if (s_mr >= 0) or (v[r] == 0) or (v[obj] == 0):
        return inf  # type: ignore
    return floor(
        log(1 - ((w[m] * v["Growth"]) / (s_mr * v[r] * biomass))) / (v["Growth"] * tau)
    )


def estimate_mn_state_duration_ts(
    mn: MetabolicNetwork,
    obj: str,
    biomass: float,
    tau: float,
    w: dict[str, float],
    v: dict[str, float],
) -> int:
    return min(
        estimate_metabolite_depletion_ts(mn, obj, m, tau, biomass, w, v) for m in w
    )


def estimate_state_duration(
    mn: MetabolicNetwork,
    bn: BooleanNetwork | None,
    tau: float,
    biomass: float,
    x: dict[str, int],
    v: dict[str, float],
    w: dict[str, float],
    obj: str,
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
    x_plus_1: dict[str, int] = {}
    if bn is not None:
        x_plus_1 = update_regulatory_state(
            mn,
            bn,
            w_plus_1,
            v,
            x,
        )
    if x_plus_1 != x:
        return 1
    return max(1, estimate_mn_state_duration_ts(mn, obj, biomass, tau, w, v))


# ==============================================================================
# Main Functions
# ==============================================================================
def next_iter(
    mn: MetabolicNetwork,
    bn: BooleanNetwork | None,
    fba: FluxBalanceAnalysis,
    obj: str,
    tau: float,
    v: dict[str, float],
    w: dict[str, float],
    x: dict[str, int],
    biomass: float,
    duration: int,
) -> tuple[dict[str, float], dict[str, float], dict[str, int], float, int]:
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
    next_x: dict[str, int] = {}

    bounds: dict[str, tuple[float, float]] = update_bounds(
        mn,
        tau,
        next_biomass,
        next_w,
    )
    if bn is not None:
        next_x = update_regulatory_state(mn, bn, next_w, v, x)
        bounds |= {n: (0.0, 0.0) for n in bn if n in mn.reactions() and next_x[n] == 0}
    _, next_v = fba.solve(bounds)

    next_duration: int = estimate_state_duration(
        mn,
        bn,
        tau,
        next_biomass,
        next_x,
        next_v,
        next_w,
        obj,
    )

    return (next_v, next_w, next_x, next_biomass, next_duration)


def simulate_rfba(
    sbml: str | MetabolicNetwork,
    obj: str,
    bnet: str | BooleanNetwork | None = None,
    concentrations: dict[str, float] = {},
    state: dict[str, int] = {},
    bounds: dict[str, tuple[float, float]] = {},
    mutations: dict[str, int] = {},
    biomass: float = 1e-2,
    tau: float = 1e-2,
    iter: int = 100,
    compressed: bool = False,
    lpsolver: str = DEFAULT_SOLVER,
) -> pd.DataFrame:
    # --------------------------------------------------------------------------
    # Pre-processing
    # --------------------------------------------------------------------------
    # ~ Inputs
    if isinstance(sbml, MetabolicNetwork):
        mn: MetabolicNetwork = sbml
    else:
        mn: MetabolicNetwork = MetabolicNetwork.read_sbml(sbml)
    if isinstance(bnet, BooleanNetwork):
        bn: BooleanNetwork | None = bnet
    elif bnet is not None:
        bn: BooleanNetwork | None = BooleanNetwork.load(bnet)
    else:
        bn: BooleanNetwork | None = None
    fba: FluxBalanceAnalysis = mn.instantiate_fba(obj, lpsolver=lpsolver)

    # ~ Experiment bounds
    for r, (lb, ub) in bounds.items():
        mn.set_bound(r, lb, ub)

    if len(mutations) != 0 and bn is None:
        bn = BooleanNetwork()

    if bn is not None:
        # ~ Add gene association rules and missing genes to BN
        for gene, rule in mn.genes_association().items():
            if rule != "":
                bn[gene] = rule
        for gene in mn.genes():
            if gene not in bn:
                bn[gene] = 1

        # ~ Experiment mutations
        for n, val in mutations.items():
            bn[n] = f"{val}"

    # --------------------------------------------------------------------------
    # Simulation
    # --------------------------------------------------------------------------
    v: dict[str, float] = {r: 0.0 for r in mn.reactions()}
    w: dict[str, float] = concentrations.copy()
    if bn is not None:
        x: dict[str, int] = bn(
            {n: 0 for n in bn.keys() if n not in mn.metabolites(True, False)}
            | {n: 1 if w.get(n, 0) > 0 else 0 for n in mn.metabolites(True, False)}
            | state
        )
    else:
        x: dict[str, int] = {}

    simulation_states: list[
        tuple[
            dict[str, float],
            dict[str, float],
            dict[str, int],
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
    react_cols: dict[str, str] = {
        r: f"{r} {TAG_REACTION}" for r in mn.reactions()
    }
    reg_cols: dict[str, str] = {}
    if bn is not None:
        reg_cols |= {
            n: f"{n} {TAG_REGULATION}" for n in bn if n not in mn.metabolites()
        }

    sim: list[dict[str, float | int]] = []
    iter_: int = 0
    for s in range(0, len(simulation_states)):
        v_0, w_0, x_0, biomass_0, duration_0 = simulation_states[s]
        growth_0 = v_0[obj]
        for i in range(duration_0) if not compressed else {0, duration_0 - 1}:
            time: float = (iter_ + i) * tau
            time_rounded: float = round(time, abs(floor(log(tau, 10))))
            biomass_i: float = update_biomass(tau, biomass_0, growth_0, i + 1)
            w_i: dict[str, float] = update_kinetics(
                mn, tau, biomass_0, growth_0, w_0, v_0, i + 1
            )
            state_i: dict[str, float | int] = (
                {
                    "Time": time_rounded,
                    "biomass": biomass_i,
                }
                | {met_cols[m]: w_i[m] for m in w_i}
                | {react_cols[r]: v_0[r] for r in v_0}
                | {reg_cols[n]: x_0[n] for n in x_0 if n not in mn.metabolites()}
            )
            sim.append(state_i)

        iter_ += duration_0

    cols: list[str] = [
        "Time",
        *sorted(met_cols.values()),
        *sorted(react_cols.values()),
        *sorted(reg_cols.values()),
        "biomass",
    ]
    sim_df: pd.DataFrame = pd.DataFrame(sim, columns=cols)

    return sim_df


# ==============================================================================
# Main
# ==============================================================================
if __name__ == "__main__":
    ...
