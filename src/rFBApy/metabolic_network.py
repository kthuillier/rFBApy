# ==============================================================================
# Imports
# ==============================================================================
from __future__ import annotations

import os

from libsbml import (  # type: ignore
    FbcAssociation,
    FbcModelPlugin,
    FbcReactionPlugin,
    GeneProductAssociation,
    Model,
    Reaction,
    SBMLDocument,
    SBMLReader,
)

from rFBApy.asp import asp_number, asp_string, write_asp
from rFBApy.fba import (
    DEFAULT_SOLVER,
    FluxBalanceAnalysis,
    GlpkFba,
    GurobiFba,
)


# ==============================================================================
# Metabolic Network
# ==============================================================================
# > Data structure use to model Metabolic Network
# > Associated with all the basic functions needed for Flux Balance Analysis
# ==============================================================================
class MetabolicNetwork:
    # ==========================================================================
    # Initialization
    # ==========================================================================
    def __init__(self: MetabolicNetwork):
        self.irreversible: bool = True

        # ----------------------------------------------------------------------
        # Metabolism
        # ----------------------------------------------------------------------
        self.__metabolites: set[str] = set()
        self.__reactions: set[str] = set()
        self.__exchanges: dict[str, str | None] = {}
        self.__S: dict[tuple[str, str], float] = {}
        self.__bounds: dict[str, tuple[float, float]] = {}
        self.__reversible_reactions: set[str] = set()

        # ----------------------------------------------------------------------
        # Genes
        # ----------------------------------------------------------------------
        self.__genes: set[str] = set()
        self.__genes_association: dict[str, str] = {}

        # ----------------------------------------------------------------------
        # Reversible reaction cache
        # ----------------------------------------------------------------------
        self.__reversible_reactions_mapping: dict[str, tuple[str, str]] = {}

        # ----------------------------------------------------------------------
        # External metabolites cache (structure is immutable once built, only
        # bounds change afterwards, so this is safe to memoize)
        # ----------------------------------------------------------------------
        self.__external_metabolites: set[str] | None = None

    # ==========================================================================
    # Getters
    # ==========================================================================
    def reactions(self: MetabolicNetwork, only_exchange: bool = False) -> set[str]:
        if only_exchange:
            return set(self.__exchanges.values())  # type: ignore
        return self.__reactions

    def previously_reversible_reactions(
        self: MetabolicNetwork,
    ) -> dict[str, tuple[str, str]]:
        return self.__reversible_reactions_mapping

    def metabolites(
        self: MetabolicNetwork, external: bool = True, internal: bool = True
    ) -> set[str]:
        if external and internal:
            return self.__metabolites | self.__external_metabolites_cache()
        if external:
            return self.__external_metabolites_cache()
        if internal:
            return self.__metabolites
        return set()

    def __external_metabolites_cache(self: MetabolicNetwork) -> set[str]:
        if self.__external_metabolites is None:
            self.__external_metabolites = set(self.__exchanges.keys())
        return self.__external_metabolites

    def exchange(self: MetabolicNetwork, metabolite: str) -> str | None:
        assert metabolite in self.__exchanges
        return self.__exchanges[metabolite]

    def bound(self: MetabolicNetwork, r: str) -> tuple[float, float]:
        assert r in self.__bounds
        return self.__bounds[r]

    def bounds(self: MetabolicNetwork) -> dict[str, tuple[float, float]]:
        return self.__bounds

    def stoichiometry(self: MetabolicNetwork) -> dict[tuple[str, str], float]:
        return self.__S

    def is_reversible(self: MetabolicNetwork, r: str) -> bool:
        return r in self.__reversible_reactions

    def genes(self: MetabolicNetwork) -> set[str]:
        return self.__genes

    def genes_association(self: MetabolicNetwork) -> dict[str, str]:
        return self.__genes_association

    # ==========================================================================
    # Setters
    # ==========================================================================
    def set_bound(self: MetabolicNetwork, r: str, lb: float, ub: float) -> None:
        for r_ in self.__reversible_reactions_mapping.get(r, (r,)):
            assert r_ in self.__bounds
            self.__bounds[r_] = (lb, ub)

    # ==========================================================================
    # Auxiliary functions
    # ==========================================================================
    def copy(self: MetabolicNetwork) -> MetabolicNetwork:
        mn_: MetabolicNetwork = MetabolicNetwork()
        mn_.irreversible = self.irreversible
        mn_.__metabolites = self.__metabolites.copy()
        mn_.__reactions = self.__reactions.copy()
        mn_.__exchanges = self.__exchanges.copy()
        mn_.__S = self.__S.copy()
        mn_.__bounds = self.__bounds.copy()
        mn_.__reversible_reactions = self.__reversible_reactions.copy()
        mn_.__genes = self.__genes.copy()
        mn_.__genes_association = self.__genes_association.copy()
        mn_.__reversible_reactions_mapping = self.__reversible_reactions_mapping.copy()
        return mn_

    def to_irreversible(self: MetabolicNetwork) -> MetabolicNetwork:
        def __irreversible_renaming(r: str) -> tuple[str, str]:
            return f"{r}_forward", f"{r}_reverse"

        mn_: MetabolicNetwork = self.copy()
        mn_.irreversible = True
        # ----------------------------------------------------------------------
        # Update Exchange reactions
        # ----------------------------------------------------------------------
        for m, r in self.__exchanges.items():
            assert r is not None
            if not self.is_reversible(r):
                continue
            rf, rr = __irreversible_renaming(r)
            mn_.__reversible_reactions_mapping[r] = (rf, rr)
            if self.__S[(m, r)] < 0:  # case: m is a reactant
                mn_.__exchanges[m] = rf
            else:  # case: m is a product
                mn_.__exchanges[m] = rr
        # ----------------------------------------------------------------------
        # Update Reactions
        # ----------------------------------------------------------------------
        for r in self.__reactions:
            if not self.is_reversible(r):
                continue
            rf, rr = __irreversible_renaming(r)
            mn_.__reversible_reactions_mapping[r] = (rf, rr)
            # ------------------------------------------------------------------
            # Update reaction set
            # ------------------------------------------------------------------
            mn_.__reactions.remove(r)
            mn_.__reactions.add(rf)
            mn_.__reactions.add(rr)
            # ------------------------------------------------------------------
            # Update bounds
            # ------------------------------------------------------------------
            del mn_.__bounds[r]
            mn_.__bounds[rf] = (0, self.__bounds[r][1])
            mn_.__bounds[rr] = (0, -self.__bounds[r][0])
            # ------------------------------------------------------------------
            # Update stoichiometry
            # ------------------------------------------------------------------
            for m, r_ in self.__S:
                if r != r_:
                    continue
                del mn_.__S[(m, r)]
                mn_.__S[(m, rf)] = self.__S[(m, r)]
                mn_.__S[(m, rr)] = -self.__S[(m, r)]
            # ------------------------------------------------------------------
            # Update gene association
            # ------------------------------------------------------------------
            del mn_.__genes_association[r]
            mn_.__genes_association[rf] = self.__genes_association[r]
            mn_.__genes_association[rr] = self.__genes_association[r]
        # ----------------------------------------------------------------------
        # Return
        # ----------------------------------------------------------------------
        return mn_

    # ==========================================================================
    # Parsing
    # ==========================================================================
    @classmethod
    def __parse_gene_formula_rec(cls, assoc: FbcAssociation) -> str:
        # ~ Walk the FbcAssociation object tree directly (FbcAnd/FbcOr/
        # GeneProductRef), rather than going through
        # GeneProductAssociation.toXMLNode() (see __parse_gene_formula) --
        # both give identical results (verified against every reaction of
        # iJR904 and iML1515), but toXMLNode() rebuilds and leaks a whole new
        # XML subtree on every single call: its result comes back with
        # `thisown = False`, so libsbml's SWIG bindings never free it, no
        # matter what Python does with the wrapper. Called once per
        # gene-associated reaction in every read_sbml() call - on a
        # genome-scale model reparsed per condition in a large batch, that
        # leak alone was enough to grow each worker by ~1.5MB *per
        # condition*, unbounded (confirmed by isolating read_sbml() in a
        # loop: RSS climbed steadily while gc-tracked object count stayed
        # flat - i.e. native memory invisible to Python's own GC).
        if assoc.isGeneProductRef():
            return assoc.getGeneProduct() # pyright: ignore[reportAttributeAccessIssue]
        children: list[str] = [
            cls.__parse_gene_formula_rec(child)
            for child in assoc.getListOfAssociations() # pyright: ignore[reportAttributeAccessIssue]
        ]
        if len(children) == 1:
            return children[0]
        if assoc.isFbcAnd():
            return "(" + " & ".join(children) + ")"
        elif assoc.isFbcOr():
            return "(" + " | ".join(children) + ")"
        print("Unknown fbc operator")
        assert False

    @classmethod
    def __parse_gene_formula(cls, r: Reaction) -> str:
        r_fbc: FbcReactionPlugin | None = r.getPlugin("fbc")
        if r_fbc is None:
            return ""
        gpa: GeneProductAssociation = r_fbc.getGeneProductAssociation()
        if gpa is None:
            return ""
        assoc: FbcAssociation | None = gpa.getAssociation()
        if assoc is None:
            return ""
        return cls.__parse_gene_formula_rec(assoc)

    @classmethod
    def read_sbml(cls, sbml_file: str) -> MetabolicNetwork:
        name: str

        if not os.path.isfile(sbml_file):
            raise FileNotFoundError(
                f"SBML metabolic network file not found: {sbml_file}"
            )

        # Object initialization
        mn: MetabolicNetwork = cls()

        # Open SBML file
        sbmld: SBMLDocument = SBMLReader().readSBML(sbml_file)  # type: ignore
        sbmlm: Model = sbmld.getModel() # pyright: ignore[reportAssignmentType]
        sbml_fbc: FbcModelPlugin = sbmlm.getPlugin("fbc")

        # Param
        param: dict[str, float] = {}
        for parameter in sbmlm.getListOfParameters():
            param[parameter.getId()] = parameter.getValue()

        # Genes
        if sbml_fbc is not None:
            for gene_product in sbml_fbc.getListOfGeneProducts():
                mn.__genes.add(gene_product.getId())

        # Species
        for species in sbmlm.getListOfSpecies():
            name = species.getId()
            if not species.getBoundaryCondition():
                mn.__metabolites.add(name)
            else:
                mn.__exchanges[name] = None

        # Reactions
        for reaction in sbmlm.getListOfReactions():
            name = reaction.getId()
            mn.__reactions.add(name)
            stoechiometry: float
            # Reactants
            for reactant in reaction.getListOfReactants():
                stoechiometry = float(reactant.getStoichiometry())
                reactant_name: str = reactant.getSpecies()
                mn.__S[(reactant_name, name)] = -stoechiometry
                if reactant_name in mn.__exchanges:
                    mn.__exchanges[reactant_name] = name

            # Products
            for product in reaction.getListOfProducts():
                stoechiometry = float(product.getStoichiometry())
                product_name: str = product.getSpecies()
                mn.__S[(product_name, name)] = stoechiometry
                if product_name in mn.__exchanges:
                    mn.__exchanges[product_name] = name

            # Bounds
            reaction_fbc: FbcReactionPlugin | None = reaction.getPlugin("fbc")
            low_bound: float
            up_bound: float
            if reaction_fbc is not None:
                low_bound = param[reaction_fbc.getLowerFluxBound()]
                up_bound = param[reaction_fbc.getUpperFluxBound()]
            else:
                low_bound = 0
                up_bound = float("inf")
                if reaction.getReversible():
                    low_bound = float("-inf")
            mn.__bounds[name] = (low_bound, up_bound)

            # Is Reversible
            if reaction.getReversible():
                mn.__reversible_reactions.add(name)
                mn.irreversible = False

            # Genes formula
            mn.__genes_association[name] = MetabolicNetwork.__parse_gene_formula(
                reaction
            )

        return mn

    # ==========================================================================
    # Convert
    # ==========================================================================
    def to_asp(self: MetabolicNetwork, filename: str | None = None) -> str:
        # ~ reaction/3, bounds/3 and ext/2 facts of fixpoint/asp/fixpoint.lp
        lines: list[str] = [
            f"% Metabolic network: {len(self.__reactions)} reactions, "  # noqa: ISC004
            f"{len(self.__metabolites)} internal and "
            f"{len(self.__exchanges)} external metabolites"
        ]
        for (m, r), coeff in self.__S.items():
            lines.append(
                f"reaction({asp_string(r)},{asp_string(m)},{asp_number(coeff)})."
            )
        for r, (lb, ub) in self.__bounds.items():
            lines.append(f"bounds({asp_string(r)},{asp_number(lb)},{asp_number(ub)}).")
        for m, r in self.__exchanges.items():
            if r is not None:
                lines.append(f"ext({asp_string(m)},{asp_string(r)}).")
        return write_asp(lines, filename)

    def instantiate_fba(
        self: MetabolicNetwork,
        objective: str,
        lpsolver: str = DEFAULT_SOLVER,
    ) -> FluxBalanceAnalysis:
        if objective not in self.__reactions:
            raise ValueError(
                f"Objective reaction '{objective}' not found in the "
                "metabolic network."
            )
        stoichiometry: dict[tuple[str, str], float] = {
            (m, r): coeff
            for (m, r), coeff in self.stoichiometry().items()
            if m not in self.__exchanges
        }
        if lpsolver == "glpk":
            return GlpkFba.load_mn(objective, stoichiometry, self.__bounds.copy())
        if lpsolver == "gurobi":
            return GurobiFba.load_mn(objective, stoichiometry, self.__bounds.copy())
        else:
            raise ValueError(f"Unknown LP Solver: {lpsolver}.")
