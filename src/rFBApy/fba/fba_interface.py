# ==============================================================================
# Imports
# ==============================================================================
from __future__ import annotations

from typing import Literal

# ~ Float solvers: values within this of 0 are numerical noise and returned as 0
ACCURACY: float = 1e-8


# ==============================================================================
# Flux Balance Analysis - Interface
# ==============================================================================
class FluxBalanceAnalysis:

    def __init__(self: FluxBalanceAnalysis) -> None:
        self._bounds: dict[str, tuple[float, float]] = {}
        self._objective: str = ""
        # ~ Clipping of the values returned by solve(); None: no clipping
        self._lpeps: float | None = ACCURACY
        # ~ pFBA: reaction -> auxiliary |flux| variable, and the two
        # objective-coefficient vectors switched between the growth
        # maximization step and the total-flux minimization step
        self.__pfba_var: dict[str, str] = {}
        self.__pfba_aux_vars: frozenset[str] = frozenset()
        self.__pfba_coeffs: dict[str, float] = {}
        self.__fba_coeffs: dict[str, float] = {}
        # ~ Reactions to build |flux| auxiliary variables for, *if* pFBA ever
        # gets requested -- see __setup_pfba().
        self.__pfba_reactions: frozenset[str] = frozenset()

    # --------------------------------------------------------------------------
    # Initialization from SBML file
    # --------------------------------------------------------------------------
    @classmethod
    def load_sbml(cls: type[FluxBalanceAnalysis], sbml: str, objective: str,
                  lpeps: float | None = ACCURACY) -> FluxBalanceAnalysis:
        # ~ Lazy: metabolic_network imports this module
        from rFBApy.metabolic_network import MetabolicNetwork
        mn: MetabolicNetwork = MetabolicNetwork.read_sbml(sbml)
        internal: set[str] = mn.metabolites(False, True)
        return cls.load_mn(
            objective,
            {(m, r): c for (m, r), c in mn.stoichiometry().items() if m in internal},
            mn.bounds(),
            lpeps,
        )

    # --------------------------------------------------------------------------
    # Initialization from Metabolic Network
    # --------------------------------------------------------------------------
    @classmethod
    def load_mn(cls: type[FluxBalanceAnalysis],
                objective: str,
                stoichiometry: dict[tuple[str, str], float],
                bounds: dict[str, tuple[float, float]],
                lpeps: float | None = ACCURACY) \
            -> FluxBalanceAnalysis:
        # Adapt the stoichiometry matrix to the needed format
        stoichiometry_: dict[str, set[tuple[float, str]]] = {}
        for (m, r), coeff in stoichiometry.items():
            stoichiometry_.setdefault(m, set()).add((coeff, r))
        # Initialize the FBA object
        fba: FluxBalanceAnalysis = cls._lpinit(
            objective, stoichiometry_, bounds
        )
        fba._objective = objective
        fba._lpeps = lpeps
        # ~ pFBA setup is deferred to the first
        # solve(pfba=True) call, not done unconditionally here.
        fba.__pfba_reactions = frozenset(bounds.keys())
        # Return the initialized FBA object
        return fba

    # --------------------------------------------------------------------------
    # Setters
    # --------------------------------------------------------------------------
    # ~ Init
    @classmethod
    def _lpinit(cls: type[FluxBalanceAnalysis], objective: str,
                stoichiometry: dict[str, set[tuple[float, str]]],
                bounds: dict[str, tuple[float, float]]) \
            -> FluxBalanceAnalysis:
        raise NotImplementedError()

    # ~ Bounds
    def __reset_bounds(self: FluxBalanceAnalysis,
                       r: str | None = None) -> None:
        if r is None:
            self.__set_bounds(self._bounds)
        else:
            self._set_bound(r, self._bounds[r][0], self._bounds[r][1])

    def __set_bounds(self: FluxBalanceAnalysis,
                     bounds: dict[str, tuple[float, float]]) -> None:
        for r, (lb, ub) in bounds.items():
            self._set_bound(r, lb, ub)

    def _set_bound(self: FluxBalanceAnalysis, r: str, lb: float, ub: float) \
            -> None:
        raise NotImplementedError()

    # ~ Model structure (used to build the pFBA auxiliary |flux| variables)
    def _add_variable(self: FluxBalanceAnalysis, r: str, lb: float, ub: float) \
            -> None:
        raise NotImplementedError()

    def _add_constraint(
        self: FluxBalanceAnalysis,
        name: str,
        expr: set[tuple[float, str]],
        lb: float,
        ub: float,
    ) -> None:
        raise NotImplementedError()

    def _set_objective(
        self: FluxBalanceAnalysis,
        coeffs: dict[str, float],
        sense: Literal['max', 'min'],
    ) -> None:
        raise NotImplementedError()

    # --------------------------------------------------------------------------
    # Parsimonious FBA setup
    # --------------------------------------------------------------------------
    # > For every reaction r, add an auxiliary variable a_r >= |v_r| (encoded as
    # > a_r >= v_r and a_r >= -v_r) so that minimizing sum(a_r) minimizes the
    # > total absolute flux, regardless of whether r is reversible.
    def __setup_pfba(self: FluxBalanceAnalysis, reactions: frozenset[str]) -> None:
        if self.__pfba_var:
            # Already built (e.g. a previous solve(pfba=True) call on this
            # same fba instance -- rfba.py reuses one fba object across every
            # time step of a simulation) -- nothing to do.
            return
        self.__fba_coeffs = {self._objective: 1.0}
        self.__pfba_coeffs = {}
        for r in sorted(reactions):
            a_r: str = f"pfba_abs_{r}"
            self.__pfba_var[r] = a_r
            self._add_variable(a_r, 0.0, float('inf'))
            self._add_constraint(f"pfba_pos_{r}", {(1.0, a_r), (-1.0, r)}, 0.0, float('inf'))
            self._add_constraint(f"pfba_neg_{r}", {(1.0, a_r), (1.0, r)}, 0.0, float('inf'))
            self.__pfba_coeffs[a_r] = 1.0
        self.__pfba_aux_vars = frozenset(self.__pfba_var.values())

    # --------------------------------------------------------------------------
    # Solving
    # --------------------------------------------------------------------------
    # > Two-step parsimonious FBA: (i) maximize the objective reaction, then
    # > (ii) fix it to its optimum and minimize the total absolute flux over
    # > every reaction, so the returned flux vector is the most parsimonious
    # > one achieving the optimal objective value.
    # >
    # > Returns (opt, fluxes): opt is None iff the LP is infeasible (then fluxes
    # > is {}), inf if unbounded, else the optimum of the objective reaction;
    # > fluxes has every reaction (no auxiliary variable). Values are clean: each
    # > backend removes its own numerical noise (float backends clip to 0 what
    # > is within lpeps of it, see _lpeps), so callers use them as they are.
    def solve(
        self: FluxBalanceAnalysis,
        bounds: dict[str, tuple[float, float]] | None = None,
        pfba: bool = True,
    ) -> tuple[None | float, dict[str, float]]:
        if bounds is None:
            bounds = {}
        self.__set_bounds(bounds)
        # ~ exact=False when pFBA is about to follow: phase 2 re-solves with
        # the objective reaction fixed (via an equality bound) to this very
        # value, and an exact-arithmetic refinement of *this* solve can find
        # that a floating-point-computed value doesn't exactly satisfy the
        # rational constraint system, spuriously flipping the fixed-bound
        # phase-2 problem to infeasible. Only the final solve of the chain
        # should request the exact-arithmetic pass.
        opt: None | float = self._lpsolve(exact=not pfba)
        lp_state: dict[str, float] = {}
        if opt is not None:
            lp_state = self.__reaction_state()
            if pfba and opt not in (float('inf'), float('-inf')):
                # ~ Lazy, idempotent: only pays for the |flux| auxiliary
                # variables/constraints the first time pFBA is actually used.
                self.__setup_pfba(self.__pfba_reactions)
                self._set_bound(self._objective, opt, opt)
                self._set_objective(self.__pfba_coeffs, 'min')
                # ~ warm=True: only the objective row and one reaction's bound
                # changed since the phase-1 solve, so the phase-1 optimal
                # basis is reused instead of paying for a full cold restart.
                # exact=False for the same reason as above (fixed-bound
                # equality constraint derived from a floating-point optimum).
                pfba_opt: None | float = self._lpsolve(warm=True, exact=False)
                if pfba_opt is not None:
                    lp_state = self.__reaction_state()
                self._set_objective(self.__fba_coeffs, 'max')
        self.__reset_bounds()
        if opt is None or self._lpeps is None:
            return opt, lp_state
        return self.__clip(opt, self._lpeps), {
            r: self.__clip(v, self._lpeps) for r, v in lp_state.items()
        }

    @staticmethod
    def __clip(value: float, lpeps: float) -> float:
        return value if abs(value) > lpeps else 0.0

    def __reaction_state(self: FluxBalanceAnalysis) -> dict[str, float]:
        return {
            r: v for r, v in self._lpstate().items() if r not in self.__pfba_aux_vars
        }

    def _lpsolve(
        self: FluxBalanceAnalysis, warm: bool = False, exact: bool = True
    ) -> None | float:
        raise NotImplementedError()

    def _lpstate(self: FluxBalanceAnalysis) -> dict[str, float]:
        raise NotImplementedError()