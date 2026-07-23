# ==============================================================================
# Imports
# ==============================================================================
from __future__ import annotations

from typing import Literal

from libsbml import (  # type: ignore
    Model,
    SBMLDocument,
    SBMLReader,
    FbcReactionPlugin
)


# ==============================================================================
# Flux Balance Analysis - Interface
# ==============================================================================
class FluxBalanceAnalysis:

    def __init__(self: FluxBalanceAnalysis) -> None:
        self._bounds: dict[str, tuple[float, float]] = {}
        self._objective: str = ""
        # ~ pFBA: reaction -> auxiliary |flux| variable, and the two
        # objective-coefficient vectors switched between the growth
        # maximization step and the total-flux minimization step
        self.__pfba_var: dict[str, str] = {}
        self.__pfba_aux_vars: frozenset[str] = frozenset()
        self.__pfba_coeffs: dict[str, float] = {}
        self.__fba_coeffs: dict[str, float] = {}

    # --------------------------------------------------------------------------
    # Initialization from SBML file
    # --------------------------------------------------------------------------
    @classmethod
    def __read_sbml(cls: type[FluxBalanceAnalysis], sbml: str) \
            -> tuple[dict[str, set[tuple[float, str]]],
                     dict[str, tuple[float, float]]]:
        # Object initialization
        stochiometric_matrix: dict[str, set[tuple[float, str]]] = {}
        bounds: dict[str, tuple[float, float]] = {}

        # Open SBML file
        sbmld: SBMLDocument = SBMLReader().readSBML(sbml)  # type: ignore
        sbmlm: Model = sbmld.getModel()  # type: ignore

        # Param
        param: dict[str, float] = {}
        for parameter in sbmlm.getListOfParameters():
            param[parameter.getId()] = parameter.getValue()

        name: str
        # Species
        boundary_species: dict[str, tuple[float, str]] = {}
        for species in sbmlm.getListOfSpecies():
            name = species.getId()
            if species.getBoundaryCondition():
                boundary_species[name] = (0, '')

        # Reactions
        for reaction in sbmlm.getListOfReactions():
            name = reaction.getId()
            stoechiometry: float
            # Reactants
            for reactant in reaction.getListOfReactants():
                stoechiometry = float(reactant.getStoichiometry())
                reactant_name: str = reactant.getSpecies()
                if reactant_name in boundary_species:
                    assert boundary_species[reactant_name][0] == 0
                    boundary_species[reactant_name] = (-stoechiometry, name)
                    continue
                stochiometric_matrix.setdefault(
                    reactant_name, set()
                ).add((-stoechiometry, name))

            # Products
            for product in reaction.getListOfProducts():
                stoechiometry = float(product.getStoichiometry())
                product_name: str = product.getSpecies()
                if product_name in boundary_species:
                    assert boundary_species[product_name][0] == 0
                    boundary_species[product_name] = (stoechiometry, name)
                    continue
                stochiometric_matrix.setdefault(
                    product_name, set()
                ).add((stoechiometry, name))

            # Bounds
            reaction_fbc: FbcReactionPlugin | None = reaction.getPlugin('fbc')
            bounds[name] = (float('-inf'), float('inf'))
            if reaction_fbc is not None:
                bounds[name] = (
                    param[reaction_fbc.getLowerFluxBound()],
                    param[reaction_fbc.getUpperFluxBound()]
                )
            else:
                if not reaction.getReversible():
                    bounds[name] = (0, float('inf'))

        return (stochiometric_matrix, bounds)

    # --------------------------------------------------------------------------
    # Initialization from SBML file
    # --------------------------------------------------------------------------
    @classmethod
    def load_sbml(cls: type[FluxBalanceAnalysis], sbml: str, objective: str) \
            -> FluxBalanceAnalysis:
        # Parse SBML
        stoichiometry, bounds = cls.__read_sbml(sbml)
        # Initialize the FBA object
        fba: FluxBalanceAnalysis = cls._lpinit(
            objective, stoichiometry, bounds
        )
        fba._objective = objective
        fba.__setup_pfba(set(bounds.keys()))
        # Return the initialized FBA object
        return fba

    # --------------------------------------------------------------------------
    # Initialization from Metabolic Network
    # --------------------------------------------------------------------------
    @classmethod
    def load_mn(cls: type[FluxBalanceAnalysis],
                objective: str,
                stoichiometry: dict[tuple[str, str], float],
                bounds: dict[str, tuple[float, float]]) \
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
        fba.__setup_pfba(set(bounds.keys()))
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
    def __setup_pfba(self: FluxBalanceAnalysis, reactions: set[str]) -> None:
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
    def solve(
        self: FluxBalanceAnalysis,
        bounds: dict[str, tuple[float, float]] = {},
        pfba: bool = True,
    ) -> tuple[None | float, dict[str, float]]:
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
        return opt, lp_state

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