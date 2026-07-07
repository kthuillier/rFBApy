# ==============================================================================
# Imports
# ==============================================================================
from __future__ import annotations

from typing import Any, Literal

# ~ GLPK
from swiglpk import (  # type: ignore
    GLP_CV,
    GLP_DB,
    GLP_FEAS,
    GLP_FR,
    GLP_INFEAS,
    GLP_LO,
    GLP_MAX,
    GLP_OFF,
    GLP_OPT,
    GLP_FX,
    GLP_SF_AUTO,
    GLP_UNBND,
    GLP_UP,
    doubleArray,
    glp_add_cols,
    glp_add_rows,
    glp_adv_basis,
    glp_create_index,
    glp_create_prob,
    glp_get_col_prim,
    glp_get_num_cols,
    glp_get_num_rows,
    glp_get_obj_val,
    glp_get_status,
    glp_init_smcp,
    glp_scale_prob,
    glp_set_col_bnds,
    glp_set_col_kind,
    glp_set_col_name,
    glp_set_mat_row,
    glp_set_obj_coef,
    glp_set_obj_dir,
    glp_set_prob_name,
    glp_set_row_bnds,
    glp_set_row_name,
    glp_simplex,
    glp_unscale_prob,
    glp_std_basis,
    glp_smcp,
    glp_term_out,
    intArray,
)

from rFBApy.fba.fba_interface import FluxBalanceAnalysis

GLPK_TOL: float = 1e-8

# ==============================================================================
# Flux Balance Analysis - GLPK Implementation
# ==============================================================================

class GlpkFba(FluxBalanceAnalysis):

    def __init__(self: GlpkFba) -> None:
        super().__init__()
        # ----------------------------------------------------------------------
        # Glpk model initialisation
        # ----------------------------------------------------------------------
        # ~ Init model
        self.model: Any = glp_create_prob()
        glp_create_index(self.model)
        glp_set_prob_name(self.model, 'GlpkFBA')
        glp_set_obj_dir(self.model, GLP_MAX)
        self.__smcp = glp_smcp()
        glp_init_smcp(self.__smcp)
        # ~ Tighten tolerances (default GLPK tol_bnd/tol_dj/tol_piv are looser than Gurobi's)
        glp_term_out(GLP_OFF)

        # ----------------------------------------------------------------------
        # Memory
        # ----------------------------------------------------------------------
        self.__variables: dict[str, int] = {}

    # --------------------------------------------------------------------------
    # Setters
    # --------------------------------------------------------------------------
    # ~ Init
    @classmethod
    def _lpinit(cls: type[GlpkFba], objective: str,
                stoichiometry: dict[str, set[tuple[float, str]]],
                bounds: dict[str, tuple[float, float]]) -> GlpkFba:
        # Init Gurobi Model
        fba: GlpkFba = GlpkFba()
        index: int
        # ~ Add variables
        for r, (lb, ub) in bounds.items():
            glp_add_cols(fba.model, 1)
            index = glp_get_num_cols(fba.model)
            glp_set_col_name(fba.model, index, r)
            glp_set_col_kind(fba.model, index, GLP_CV)
            fba.__variables[r] = index
            fba._set_bound(r, lb, ub)
        fba._bounds = bounds.copy()
        # ~ Add constraints
        for m, expr in stoichiometry.items():
            # ~ Add new matrix named row
            glp_add_rows(fba.model, 1)
            index = glp_get_num_rows(fba.model)
            consname: str = f'steadystate_{m}'
            glp_set_row_name(fba.model, index, consname)
            # ~ Fill the row
            num_cols: int = glp_get_num_cols(fba.model)
            num_vars: int = len(expr)
            index_array: intArray = intArray(num_cols + 1)
            value_array: doubleArray = doubleArray(num_cols + 1)
            for i, (coeff, r) in enumerate(expr):
                assert r in fba.__variables
                varindex: int = fba.__variables[r]
                index_array[i + 1] = varindex
                value_array[i + 1] = coeff
            glp_set_mat_row(fba.model, index, num_vars, index_array, value_array)
            # ~ Set the row bounds
            glp_set_row_bnds(fba.model, index, GLP_FX, 0., 0.)
        # ~ Add objective
        glp_set_obj_coef(fba.model, fba.__variables[objective], 1.)
        return fba

    # ~ Bounds
    def _set_bound(self: GlpkFba, r: str, lb: float, ub: float) -> None:
        assert r in self.__variables
        if lb == float('-inf') and ub == float('inf'):
            glp_set_col_bnds(self.model, self.__variables[r], GLP_FR, 0., 0.)
        elif lb == float('-inf'):
            glp_set_col_bnds(self.model, self.__variables[r], GLP_UP, 0., ub)
        elif ub == float('inf'):
            glp_set_col_bnds(self.model, self.__variables[r], GLP_LO, lb, 0.)
        elif lb == ub:
            glp_set_col_bnds(self.model, self.__variables[r], GLP_FX, lb, ub)
        else:
            assert lb < ub
            glp_set_col_bnds(self.model, self.__variables[r], GLP_DB, lb, ub)

    # --------------------------------------------------------------------------
    # Solving
    # --------------------------------------------------------------------------
    def __lpsolve_glpk(self: GlpkFba) \
            -> Literal['optimal', 'infeasible', 'undefined', 'unbounded']:
        smcp = glp_smcp()
        glp_init_smcp(smcp)
        glp_std_basis(self.model)
        glp_simplex(self.model, smcp)
        glpk_status: int = glp_get_status(self.model)
        if glpk_status in [GLP_OPT, GLP_FEAS]:
            return 'optimal'
        if glpk_status in [GLP_INFEAS]:
            return 'infeasible'
        if glpk_status in [GLP_UNBND]:
            return 'unbounded'
        return 'undefined'

    def _lpsolve(self: GlpkFba) -> None | float:
        glp_unscale_prob(self.model)
        glp_scale_prob(self.model, GLP_SF_AUTO)
        glp_adv_basis(self.model, 0)   # toujours, pas seulement si 'undefined'
        status: Literal['optimal', 'infeasible', 'undefined', 'unbounded'] = \
            self.__lpsolve_glpk()
        # For GLPK: reset the basis and resolved if status is undefined
        if status == 'undefined':
            glp_adv_basis(self.model, 0)
            status = self.__lpsolve_glpk()
        # Parse solve status
        if status == 'optimal':
            return float(glp_get_obj_val(self.model))
        if status == 'unbounded':
            return float('inf')
        return None
    
    def _lpstate(self: GlpkFba) -> dict[str, float]:
        state: dict[str, float] = {}
        for r, v_idx in self.__variables.items():
            state[r] = glp_get_col_prim(self.model, v_idx)
        return state
