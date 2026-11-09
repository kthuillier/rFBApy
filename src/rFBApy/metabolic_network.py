# ==============================================================================
# Imports
# ==============================================================================
from __future__ import annotations

import xml.etree.ElementTree as ET

from rFBApy.asp import asp_number, asp_string, write_asp
from rFBApy.fba import (
    ACCURACY,
    DEFAULT_SOLVER,
    FluxBalanceAnalysis,
    GlpkFba,
    GurobiFba,
)
from rFBApy.sbml import attr, local, read_model, sections


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
        self.__all_metabolites: set[str] | None = None

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
            # ~ Cached like the other two: it is called per node and per output row
            if self.__all_metabolites is None:
                self.__all_metabolites = (
                    self.__metabolites | self.__external_metabolites_cache()
                )
            return self.__all_metabolites
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
    def __parse_gene_formula(cls, assoc: ET.Element) -> str:
        # ~ <fbc:geneProductRef>, <fbc:and> or <fbc:or> (of such elements)
        kind: str = local(assoc.tag)
        if kind == "geneProductRef":
            return attr(assoc, "geneProduct") or ""
        children: list[str] = [cls.__parse_gene_formula(c) for c in assoc]
        if len(children) == 1:
            return children[0]
        if kind == "and":
            return "(" + " & ".join(children) + ")"
        if kind == "or":
            return "(" + " | ".join(children) + ")"
        raise ValueError(f"Unknown fbc operator: {kind}")

    @classmethod
    def read_sbml(cls, sbml_file: str) -> MetabolicNetwork:
        mn: MetabolicNetwork = cls()
        model: ET.Element = read_model(sbml_file)
        lists: dict[str, ET.Element] = sections(model)
        param: dict[str, float] = {
            str(p.get("id")): float(p.get("value", "nan"))
            for p in lists.get("listOfParameters", ())
        }

        # Genes
        for gene_product in lists.get("listOfGeneProducts", ()):
            mn.__genes.add(str(attr(gene_product, "id")))

        # Species
        for species in lists.get("listOfSpecies", ()):
            name: str = str(species.get("id"))
            if species.get("boundaryCondition", "false") != "true":
                mn.__metabolites.add(name)
            else:
                mn.__exchanges[name] = None

        # Reactions
        for reaction in lists.get("listOfReactions", ()):
            name = str(reaction.get("id"))
            mn.__reactions.add(name)
            reversible: bool = reaction.get("reversible", "true") == "true"
            association: ET.Element | None = None
            for part in reaction:
                kind: str = local(part.tag)
                if kind in ("listOfReactants", "listOfProducts"):
                    sign: float = -1.0 if kind == "listOfReactants" else 1.0
                    for ref in part:
                        metabolite: str = str(ref.get("species"))
                        coeff: float = float(ref.get("stoichiometry", "1"))
                        mn.__S[(metabolite, name)] = sign * coeff
                        if metabolite in mn.__exchanges:
                            mn.__exchanges[metabolite] = name
                elif kind == "geneProductAssociation":
                    association = part[0] if len(part) else None

            # Bounds
            low_name: str | None = attr(reaction, "lowerFluxBound")
            up_name: str | None = attr(reaction, "upperFluxBound")
            if low_name is not None and up_name is not None:
                mn.__bounds[name] = (param[low_name], param[up_name])
            else:
                mn.__bounds[name] = (
                    float("-inf") if reversible else 0.0,
                    float("inf"),
                )

            # Is Reversible
            if reversible:
                mn.__reversible_reactions.add(name)
                mn.irreversible = False

            # Genes formula
            mn.__genes_association[name] = (
                "" if association is None
                else cls.__parse_gene_formula(association)
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
        lpeps: float | None = ACCURACY,
    ) -> FluxBalanceAnalysis:
        # > lpeps: float solvers return as 0 the values within lpeps of it
        # > (None: no clipping); the rational solver is exact and ignores it
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
            return GlpkFba.load_mn(
                objective, stoichiometry, self.__bounds.copy(), lpeps
            )
        if lpsolver == "gurobi":
            return GurobiFba.load_mn(
                objective, stoichiometry, self.__bounds.copy(), lpeps
            )
        if lpsolver == "rational":
            from rFBApy.fba.fba_rational import RationalFba

            return RationalFba.load_mn(
                objective, stoichiometry, self.__bounds.copy(), lpeps
            )
        raise ValueError(f"Unknown LP Solver: {lpsolver}.")
