# ==============================================================================
# Imports
# ==============================================================================
from __future__ import annotations
from rFBApy.fba.fba_interface import FluxBalanceAnalysis

# ~ Gurobi
from random import randint
from gurobipy import Model, Var, GRB, Env, LinExpr  # type: ignore

# ==============================================================================
# Flux Balance Analysis - Gurobi Implementation
# ==============================================================================

class GurobiFba(FluxBalanceAnalysis):

    def __init__(self: GurobiFba) -> None:
        super().__init__()
        # ----------------------------------------------------------------------
        # Gurobi model initialisation
        # ----------------------------------------------------------------------
        # ~ Remove all automated logs message when calling Gurobi solver
        env: Env = Env(empty=True)
        env.setParam("OutputFlag", 0)
        env.setParam("LogToConsole", 0)
        env.start()
        # ~ Init model
        self.model: Model = Model('GurobiFBA', env = env)
        self.model.setParam(GRB.Param.OutputFlag, 0)
        self.model.setParam(GRB.Param.LogToConsole, 0)
        self.model.setParam(GRB.Param.DualReductions, 0)
        self.model.setParam("Seed", randint(0, 1_000_000))
        self.model.Params.LPWarmStart = 0

        # ----------------------------------------------------------------------
        # Memory
        # ----------------------------------------------------------------------
        self.__variables: dict[str, Var] = {}

    # --------------------------------------------------------------------------
    # Setters
    # --------------------------------------------------------------------------
    # ~ Init
    @classmethod
    def _lpinit(cls: type[GurobiFba], objective: str,
                 stoichiometry: dict[str, set[tuple[float, str]]],
                 bounds: dict[str, tuple[float, float]]) -> GurobiFba:
        # Init Gurobi Model
        fba: GurobiFba = GurobiFba()
        # ~ Add variables
        for r, (lb, ub) in bounds.items():
            fba.__variables[r] = fba.model.addVar(
                name=f'f_{r}',
                vtype=GRB.CONTINUOUS,
                lb=lb,
                ub=ub
            )
        fba._bounds = bounds.copy()
        # ~ Add constraints
        for m, expr in stoichiometry.items():
            lp_expr: LinExpr = sum(
                coeff * fba.__variables[r] for coeff, r in expr  # type: ignore
            )
            fba.model.addConstr(
                lp_expr == 0,
                f'steadystate_{m}'
            )
        # ~ Add objective
        fba.model.setObjective(
            fba.__variables[objective],
            sense=GRB.MAXIMIZE
        )
        return fba

    # ~ Bounds
    def _set_bound(self: GurobiFba, r: str, lb: float, ub: float) -> None:
        assert r in self.__variables
        assert lb <= ub
        self.__variables[r].LB = lb
        self.__variables[r].UB = ub

    # --------------------------------------------------------------------------
    # Solving
    # --------------------------------------------------------------------------
    def _lpsolve(self: GurobiFba) -> None | float:
        self.model.optimize()
        status_id: int = self.model.Status
        if status_id == GRB.OPTIMAL:
            return self.model.ObjVal
        if status_id == GRB.UNBOUNDED:
            return float('inf')
        return None

    def _lpstate(self: GurobiFba) -> dict[str, float]:
        state: dict[str, float] = {}
        for r, var in self.__variables.items():
            state[r] = var.X
        return state
