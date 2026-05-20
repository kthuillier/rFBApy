from rFBApy.fba.fba_interface import FluxBalanceAnalysis
from rFBApy.fba.fba_glpk import GlpkFba
from rFBApy.fba.fba_gurobi import GurobiFba

DEFAULT_SOLVER: str = "glpk"
LP_SOLVERS: list[str] = ["gurobi", "glpk"]
