# ==============================================================================
# Import
# ==============================================================================
from __future__ import annotations

from math import exp, floor, inf, log

import pandas as pd  # type: ignore

# ~ Custom modules
from rFBApy.fba import DEFAULT_SOLVER, FluxBalanceAnalysis
from rFBApy.metabolic_network import MetabolicNetwork
from rFBApy.regulatory_network import RegulatoryNetwork

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
    bn: RegulatoryNetwork,
    w: dict[str, float],
    v: dict[str, float],
    x: dict[str, int],
    settings: dict[str, int | bool | float] = {},
) -> dict[str, int]:
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
        lb, ub = mn.bound(r)
        if s_tr < 0:
            low_bound: float = lb
            up_bound: float = min(ub, max((w[t] / (tau * biomass)), 0))
        else:
            low_bound: float = max(lb, min(-(w[t] / (tau * biomass)), 0))
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
    x: dict[str, int],
    v: dict[str, float],
    w: dict[str, float],
    obj: str,
    settings: dict[str, int | bool | float] = {},
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
    x: dict[str, int],
    biomass: float,
    duration: int,
    settings: dict[str, bool | int | float] = {},
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
        next_x = update_regulatory_state(mn, bn, next_w, v, x, settings)
        bounds |= {n: (0.0, 0.0) for n in bn if n in mn.reactions() and next_x[n] == 0}

    opt, next_v = fba.solve(bounds)
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
    concentrations: dict[str, float] = {},
    state: dict[str, int] = {},
    bounds: dict[str, tuple[float, float]] = {},
    mutations: dict[str, int] = {},
    settings: dict[str, bool | int | float] = {},
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

    if isinstance(bnet, RegulatoryNetwork):
        bn: RegulatoryNetwork | None = bnet
    elif bnet is not None:
        bn: RegulatoryNetwork | None = RegulatoryNetwork.load_bnet(bnet)
    else:
        bn: RegulatoryNetwork | None = None

    fba: FluxBalanceAnalysis = mn.instantiate_fba(obj, lpsolver=lpsolver)

    # ~ Experiment bounds
    for r, (lb, ub) in bounds.items():
        mn.set_bound(r, lb, ub)

    if (len(mutations) != 0 or len(mn.genes_association()) != 0) and bn is None:
        bn = RegulatoryNetwork()

    if bn is not None:
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
                print(f"Warning: {gene} is undefined. Set to 1.")
                bn[gene] = 1

        # ~ Experiment mutations
        for n, val in mutations.items():
            bn[n] = val

    # --------------------------------------------------------------------------
    # Simulation
    # --------------------------------------------------------------------------
    v: dict[str, float] = {r: 0.0 for r in mn.reactions()}
    w: dict[str, float] = {
        m: 0.0 for m in mn.metabolites(True, False)
    } | concentrations.copy()
    if bn is not None:
        x: dict[str, int] = bn(
            {n: 0 for n in bn if n not in mn.metabolites(True, False)}
            | {n: 1 if w.get(n, 0) > 0 else 0 for n in mn.metabolites(True, False)}
            | {n: 0 for n in mn.reactions()}
            | state  # type: ignore
            | mutations
            | settings
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
            settings=settings,
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
