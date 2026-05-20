# ==============================================================================
# Imports
# ==============================================================================
from __future__ import annotations

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
        sbmlm: Model = sbmld.getModel()

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

    # --------------------------------------------------------------------------
    # Solving
    # --------------------------------------------------------------------------
    def solve(
        self: FluxBalanceAnalysis,
        bounds: dict[str, tuple[float, float]] = {},
    ) -> tuple[None | float, dict[str, float]]:
        self.__set_bounds(bounds)
        return_value: None | float = self._lpsolve()
        lp_state: dict[str, float] = self._lpstate() if return_value is not None else {}
        self.__reset_bounds()
        return return_value, lp_state

    def _lpsolve(self: FluxBalanceAnalysis) -> None | float:
        raise NotImplementedError()

    def _lpstate(self: FluxBalanceAnalysis) -> dict[str, float]:
        raise NotImplementedError()