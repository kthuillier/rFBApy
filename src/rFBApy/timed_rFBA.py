# ==============================================================================
# Import
# ==============================================================================
from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from math import exp, floor, inf, isnan, log, log10
from pathlib import Path

import pandas as pd  # type: ignore

# ~ Custom modules
from rFBApy.rfba import (
    update_regulatory_state,
    update_biomass,
    update_kinetics
)

# ~ Custom type
PendingEvents = dict[str, tuple[int, float]]

# ==============================================================================
# Global
# ==============================================================================
ENZYME_DELAY: float = 0.25
TIME_TOLERANCE: float = 1e-8
METABOLITE_TOLERANCE: float = 1e-8
FLUX_TOLERANCE: float = 1e-8
ACCURACY: float = 1e-8

def schedule_regulatory_transitions(
    enzyme_nodes: set[str],
    x: dict[str, int | None],
    desired_x: dict[str, int | None],
    pending_events: PendingEvents,
    current_time: float,
) -> tuple[dict[str, int | None], PendingEvents]:
    x_: dict[str, int | None] = x.copy()
    pending_events_: PendingEvents = pending_events.copy()

    for key, value in desired_x.items():
        if value is None:
            continue

        if key not in enzyme_nodes:
            x_[key] = value
        else:
            if x[key] == value:
                pending_events_.pop(key, None)
                # On "annule" les modifications qui pourraient arriver
                continue
            if key in pending_events and pending_events[key][0] == value:
                continue
            pending_events_[key] = (value, current_time + ENZYME_DELAY)
    return x_, pending_events_

def initialize_simulation(
    mn: MetabolicNetwork,
    bn,
    enzyme_nodes: set[str], 
    concentrations: dict[str, float],  # noqa: B006
    state: dict[str, int],  # noqa: B006
    mutations: dict[str, int],  # noqa: B006
    settings: dict[str, bool | int | float],  # noqa: B006
    biomass: float,
) -> tuple[dict[str, float], dict[str, float], dict[str, int | None], float, float, PendingEvents]:
    v: dict[str, float] = {r: 0.0 for r in mn.reactions()}
    w: dict[str, float] = {
        m: concentrations.get(m, 0.0) for m in mn.metabolites(True, False)
    }
    pending_events = {}
    if bn is not None:
        x: dict[str, int | None] = state.copy()
        for enzyme in enzyme_nodes:
            x.setdefault(enzyme, 0)
            # Supposition faite

        desired_x: dict[str, int | None] = bn(
            {n: None for n in bn if n not in mn.metabolites(True, False)}
            | {n: 1 if w.get(n, 0) > 0 else 0 for n in mn.metabolites(True, False)}
            | {n: None for n in mn.reactions()}
            | x
            | mutations
            | settings
        )
        x, pending_events = schedule_regulatory_transitions(
            enzyme_nodes,
            x,
            desired_x,
            pending_events,
            0.0,
        )

        x |= {n: None for n in bn if n in mn.reactions()}
    else:
        x: dict[str, int | None] = {}

    return (
        v,
        w,
        x,
        biomass,
        0.0,
        pending_events,
    )

# ------------------------------------------------------------------------------
# Time Controller
# ------------------------------------------------------------------------------
def estimate_metabolite_threshold_crossing_time(
    mn: MetabolicNetwork,
    obj: str,
    m: str,
    biomass: float,
    b: float,
    w: dict[str, float],
    v: dict[str, float],
) -> float:
    if abs(w[m] - b) <= METABOLITE_TOLERANCE:
        return inf  # type: ignore  # déjà sur la borne

    r: str | None = mn.exchange(m)
    assert r is not None
    s_mr: float = mn.stoichiometry()[(m, r)]

    if abs(v[r]) <= FLUX_TOLERANCE or abs(v[obj]) <= FLUX_TOLERANCE:
        return inf  # type: ignore  # flux ou croissance nul -> pas de croisement

    direction = s_mr * v[r]  # signe de dW/dt (à un facteur biomass>0 près)

    if w[m] < b and direction <= 0:
        return inf  # type: ignore  # ne montera jamais au-dessus de b
    if w[m] > b and direction >= 0:
        return inf  # type: ignore  # ne descendra jamais sous b

    arg = 1 + ((b - w[m]) * v[obj]) / (s_mr * v[r] * biomass)
    if arg <= 0:
        return inf  # type: ignore  # asymptote : borne jamais atteinte

    dt = log(arg) / v[obj]

    if dt < -TIME_TOLERANCE:
        return inf

    return max(0.0, dt)

def estimate_next_metabolic_event(
    mn: MetabolicNetwork,
    thresholds: set[tuple[str, float]],
    obj: str,
    biomass: float,
    w: dict[str, float],
    v: dict[str, float],
) -> float:
    metabolites: set[str] = mn.metabolites(True, False)
    thresholds = thresholds.union((m, 0.0) for m in metabolites)
    return min(
        (
            estimate_metabolite_threshold_crossing_time(mn, obj, m, biomass, b, w, v)
            for m, b in thresholds
            if m in metabolites
        ),
        default=inf,
    )

def estimate_next_regulatory_event(
    pending_events: PendingEvents,
    current_time: float,
) -> float:
    return min(
        (
            max(0.0, event_time - current_time)
            for _, event_time in pending_events.values()
        ),
        default=inf,
    )

def estimate_next_event(
    mn: MetabolicNetwork,
    thresholds: set[tuple[str, float]],
    obj: str,
    biomass: float,
    w: dict[str, float],
    v: dict[str, float],
    pending_events: PendingEvents,
    current_time: float,
) -> float:

    metabolic_event_time = estimate_next_metabolic_event(
        mn,
        thresholds,
        obj,
        biomass,
        w,
        v,
    )

    regulatory_event_time = estimate_next_regulatory_event(
        pending_events,
        current_time,
    )

    return min(metabolic_event_time, regulatory_event_time)
    # Retourne le temps écoulé (pas le temps absolu)

# ------------------------------------------------------------------------------
# Update Functions
# ------------------------------------------------------------------------------
# update_regulatory_state importée depuis rfba.py

# update_biomass et update_kinetics importée depuis rfba.py 
# Elles sont réutilisées telles quelles car la fonction désirée :
# update_biomass(dt, biomass, growth) est égale à
# update_biomass(tau=dt, biomass, growth, k=1)
# On devra réimplémenter ces deux fonctions pour garder la cohérence.


def apply_due_regulatory_events(
    x: dict[str, int | None],
    pending_events: PendingEvents,
    current_time: float,
    time_tolerance: float,
) -> tuple[dict[str, int | None], PendingEvents]:
    pending_events_: PendingEvents = pending_events.copy()
    x_: dict[str, int | None] = x.copy()

    for key, (desired_state, time) in pending_events.items():
        if time <= current_time + time_tolerance:
            pending_events_.pop(key)
            x_[key] = desired_state

    return x_, pending_events_

def update_metabolic_bounds(
    mn: MetabolicNetwork,
    w: dict[str, float],
) -> dict[str, tuple[float, float]]:
    updated_bounds: dict[str, tuple[float, float]] = {}

    for m in mn.metabolites(external=True, internal=False):
        r: str | None = mn.exchange(m)
        assert r is not None
        if w[m] < -METABOLITE_TOLERANCE:
            raise ValueError(
                f"Metabolite {m} has a negative concentration: {w[m]}"
            ) # À voir si on le garde si l'erreur n'est jamais levée.

        s_mr: float = mn.stoichiometry()[(m, r)]
        lb, ub = mn.bound(r)

        if w[m] <= METABOLITE_TOLERANCE:
            if s_mr < 0:
                # Positive flux consumes the metabolite.
                ub = min(ub, 0.0)
            else:
                # Negative flux consumes the metabolite.
                lb = max(lb, 0.0)

        if lb > ub:
            raise ValueError(
                f"Depleted metabolite '{m}' requires incompatible "
                f"bounds for exchange reaction '{r}': ({lb}, {ub}). "
                "Check the original reaction bounds."
            ) # À voir si on le garde si l'erreur n'est jamais levée.
        
        updated_bounds[r] = (lb, ub)
    return updated_bounds

# ==============================================================================
# Main Functions
# ==============================================================================
def advance_to_next_event(
    mn: MetabolicNetwork,
    bn: RegulatoryNetwork | None,
    fba: FluxBalanceAnalysis,
    obj: str,
    v: dict[str, float],
    w: dict[str, float],
    x: dict[str, int | None],
    biomass: float,
    current_time: float,
    elapsed_time: float,
    pending_events: PendingEvents,
    enzyme_nodes: set[str],
    settings: dict[str, bool | int | float],  # noqa: B006,
    lpeps: float = ACCURACY,
    pfba: bool = True,
    time_tolerance: float = TIME_TOLERANCE,
) -> tuple[dict[str, float], dict[str, float], dict[str, int | None], float, float, PendingEvents]:
    next_time = current_time + elapsed_time # temps auquel l'event étudié arrive
    next_biomass: float = update_biomass(elapsed_time, biomass, v[obj], k=1)

    next_w: dict[str, float] = update_kinetics(
        mn,
        elapsed_time,
        biomass,
        v[obj],
        w,
        v,
        k=1,
    )
    
    bounds: dict[str, tuple[float, float]] = update_metabolic_bounds(mn, next_w)

    next_x: dict[str, int | None] = x.copy()
    next_pending_events: PendingEvents = pending_events.copy()

    next_x, next_pending_events = apply_due_regulatory_events(
        x,
        pending_events,
        next_time,
        time_tolerance)

    if bn is not None:
        desired_x = update_regulatory_state(mn, bn, next_w, v, next_x, settings)
        next_x, next_pending_events = schedule_regulatory_transitions(
            enzyme_nodes,
            next_x,
            desired_x,
            next_pending_events,
            next_time,
        )
        bounds |= {
            n: (0.0, 0.0)
            for n in bn
            if n in mn.reactions() and next_x[n] is not None and next_x[n] == 0
        }
    
    opt, next_v = fba.solve(bounds, pfba=pfba)
    if opt is None or opt < 0:  # FIXME
        opt = 0.0
        next_v = {r: 0.0 for r in mn.reactions()}
    else:
        opt = opt if abs(opt) > lpeps else 0.0
        next_v = {
            r: vr if abs(vr) > lpeps else 0.0
            for r, vr in next_v.items()
        }

    thresholds = bn.thresholds if bn is not None else set()
    next_elapsed_time = estimate_next_event(
        mn,
        thresholds,
        obj,
        next_biomass,
        next_w,
        next_v,
        next_pending_events,
        next_time,
    )

    return (next_v, next_w, next_x, next_biomass, next_elapsed_time, next_pending_events)
    


def simulate_timed_rfba(
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



    lpsolver: str = DEFAULT_SOLVER,


) -> pd.DataFrame:
    mn, bn = load_networks(sbml, bnet, sbmlqual)
    for r, (lb, ub) in bounds.items():
        if r in mn.reactions():
            mn.set_bound(r, lb, ub)

    fba: FluxBalanceAnalysis = mn.instantiate_fba(obj, lpsolver=lpsolver)

    bn = init_regulatory_network(mn, bn, mutations, settings)
    enzyme_nodes = set(mn.genes()) # Première supposition faite, peut-être imprécise
    
    simulation_state = initialize_simulation(
        mn, bn,
        enzyme_nodes,
        concentrations,
        state,
        mutations,
        settings,
        biomass,
    )