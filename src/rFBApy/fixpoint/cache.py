# ==============================================================================
# Imports
# ==============================================================================
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from rFBApy.fba import FluxBalanceAnalysis
from rFBApy.metabolic_network import MetabolicNetwork
from rFBApy.regulatory_network import RegulatoryNetwork

from .cut import UNUSED, Cut
from .solver import Solution

try:  # ~ No fcntl on Windows: do not share one file between processes there
    import fcntl
except ImportError:
    fcntl = None  # type: ignore

# ==============================================================================
# Global
# ==============================================================================
# > What one instance of a metabolic network leaves for the others (a batch of
# > media and strains), in a JSONL file shared by processes:
# >   - cuts, valid in any state of any instance (see cut.py);
# >   - FBA optima, keyed by objective and bounds;
# >   - fixpoints of instances without mutation, reused for a mutant whose
# >     mutations they satisfy, else used to seed its search.
# > In memory only: the FBA LPs, built once per objective and solver.
# ==============================================================================
VERSION = 1


def problem_key(objective: str, bounds: dict[str, tuple[float, float]]) -> str:
    text = objective + repr(sorted((r, (float(low), float(up))) for r, (low, up) in bounds.items()))
    return hashlib.sha1(text.encode()).hexdigest()


def fingerprint(mn: MetabolicNetwork) -> str:
    text = "|".join(f"{m},{r},{s!r}" for (m, r), s in sorted(mn.stoichiometry().items()))
    return hashlib.sha256(text.encode()).hexdigest()


# ==============================================================================
# Cache
# ==============================================================================
class Cache:
    def __init__(self: Cache, path: str | Path | None = None) -> None:
        # ~ path: loaded if it exists, then appended to; None: in memory only
        self.path: Path | None = Path(path) if path is not None else None
        self.network: str | None = None  # fingerprint of the metabolic network
        self.cuts: list[Cut] = []
        self.optima: dict[str, float | None] = {}  # problem key -> optimum
        self.fixpoints: dict[str, list[dict]] = {}  # instance key -> records
        self.stats: Counter[str] = Counter()
        self._identities: set[tuple[str, float, frozenset[tuple[str, int]]]] = set()
        self._by_problem: dict[str, Cut] = {}
        self._offset = 0  # how far the file has been read
        self._fbas: dict[tuple[str, str, bool], FluxBalanceAnalysis] = {}
        self._regulatory: dict[int, tuple[RegulatoryNetwork, str]] = {}
        self.refresh()

    # ==========================================================================
    # File
    # ==========================================================================
    def refresh(self: Cache) -> int:
        # ~ Read what this or another process appended since the last read
        if self.path is None or not self.path.exists():
            return 0
        with open(self.path, "rb") as file:
            file.seek(self._offset)
            data = file.read()
        # ~ A line still being written has no end yet: left for next time
        complete = data[: data.rfind(b"\n") + 1]
        self._offset += len(complete)
        before = len(self.cuts)
        for line in complete.decode("utf-8").splitlines():
            if line.strip():
                self._read(json.loads(line))
        return len(self.cuts) - before

    def _read(self: Cache, record: dict) -> None:
        if "cache" in record:
            if record.get("cache") != "fixpoint" or record.get("version") != VERSION:
                raise ValueError(f"{self.path} is not a fixpoint cache of version {VERSION}")
            if record["network"] is not None:
                if self.network is not None and record["network"] != self.network:
                    raise ValueError(f"{self.path} holds records of two metabolic networks")
                self.network = record["network"]
        elif "optimum" in record:
            value = record["optimum"]
            self.optima[record["problem"]] = None if value is None else float(value)
        elif "fixpoint" in record:
            stored = self.fixpoints.setdefault(record["fixpoint"], [])
            if all(other["mutations"] != record["mutations"] for other in stored):
                stored.append(record)
        elif "cut" in record:
            self._keep(
                Cut(record["objective"], float(record["cut"]), record["fluxes"], record["problem"])
            )
        else:
            raise ValueError(f"{self.path}: unknown record {sorted(record)}")

    def _append(self: Cache, record: dict) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as file:
            if fcntl is not None:
                fcntl.flock(file, fcntl.LOCK_EX)
            try:
                if file.tell() == 0:
                    header = {"cache": "fixpoint", "version": VERSION, "network": self.network}
                    file.write(json.dumps(header) + "\n")
                file.write(json.dumps(record) + "\n")
                file.flush()
            finally:
                if fcntl is not None:
                    fcntl.flock(file, fcntl.LOCK_UN)

    # ==========================================================================
    # Metabolic network
    # ==========================================================================
    def bind(self: Cache, mn: MetabolicNetwork) -> None:
        # ~ A cache belongs to one stoichiometry
        mine = fingerprint(mn)
        if self.network is None:
            self.network = mine
        elif self.network != mine:
            raise ValueError(
                "this cache was built for another metabolic network (the stoichiometry "
                "differs): its cuts do not hold here. Use one cache per metabolic network."
            )

    def fba(
        self: Cache, mn: MetabolicNetwork, objective: str, lpsolver: str, pfba: bool = False
    ) -> FluxBalanceAnalysis:
        # ~ Solved under full bounds, so one LP serves every instance. pFBA gets
        # ~ its own LP: its auxiliary variables would slow down the plain solves
        key = (objective, lpsolver, pfba)
        if key not in self._fbas:
            # ~ lpeps=None: cuts keep fluxes above their own threshold (cut.UNUSED)
            self._fbas[key] = mn.instantiate_fba(objective, lpsolver, lpeps=None)
        return self._fbas[key]

    def maximum(
        self: Cache, fba: FluxBalanceAnalysis, objective: str, bounds: dict[str, tuple[float, float]]
    ) -> tuple[str, float | None, dict[str, float] | None]:
        # ~ (problem key, optimum or None if infeasible, vector or None if cached)
        problem = problem_key(objective, bounds)
        if problem in self.optima:
            self.stats["fba reused"] += 1
            return problem, self.optima[problem], None
        optimum, vector = fba.solve(bounds, pfba=False)
        self.stats["fba solved"] += 1
        self.optima[problem] = None if optimum is None else float(optimum)
        self._append({"optimum": self.optima[problem], "problem": problem})
        return problem, self.optima[problem], vector

    # ==========================================================================
    # Cuts
    # ==========================================================================
    def _keep(self: Cache, cut: Cut) -> bool:
        if cut.identity in self._identities:
            return False
        self._identities.add(cut.identity)
        self.cuts.append(cut)
        if cut.problem:
            self._by_problem.setdefault(cut.problem, cut)
        return True

    def add(self: Cache, cut: Cut) -> None:
        if self._keep(cut):
            self.stats["cuts added"] += 1
            self._append({
                "cut": cut.growth, "objective": cut.objective, "problem": cut.problem,
                "fluxes": cut.fluxes,
            })

    def cut_of(self: Cache, problem: str) -> Cut | None:
        return self._by_problem.get(problem)

    def select(
        self: Cache, objective: str, bounds: dict[str, tuple[float, float]], limit: int
    ) -> list[Cut]:
        # ~ Cuts whose vector fits `bounds`, by decreasing growth, without those
        # ~ implied by a kept one (subset of its reactions, growth at least as high)
        closed = {(r, 1) for r, (_, upper) in bounds.items() if upper <= 0}
        closed |= {(r, -1) for r, (lower, _) in bounds.items() if lower >= 0}
        usable = [
            cut for cut in self.cuts
            if cut.objective == objective and cut.needs.isdisjoint(closed) and cut.fits(bounds)
        ]
        usable.sort(key=lambda cut: -cut.growth)
        kept: list[Cut] = []
        for cut in usable:
            if len(kept) >= limit:
                break
            if not any(other.needs <= cut.needs for other in kept):
                kept.append(cut)
        return kept

    # ==========================================================================
    # Fixpoints
    # ==========================================================================
    def regulatory_fingerprint(self: Cache, bn: RegulatoryNetwork | None) -> str:
        if bn is None:
            return ""
        known = self._regulatory.get(id(bn))
        if known is None or known[0] is not bn:
            text = "\n".join(f"{node}, {bn[node]}" for node in sorted(bn))
            known = (bn, hashlib.sha256(text.encode()).hexdigest())
            self._regulatory[id(bn)] = known
        return known[1]

    def add_fixpoint(self: Cache, instance: str, mutations: dict[str, int], solution: Solution) -> None:
        stored = self.fixpoints.setdefault(instance, [])
        mutations = {node: int(value) for node, value in mutations.items()}
        if any(other["mutations"] == mutations for other in stored):
            return
        record = {
            "fixpoint": instance,
            "mutations": mutations,
            "ones": sorted(solution.ones),
            "fluxes": {r: flux for r, flux in solution.fluxes.items() if flux != 0},
            "predicates": [[*key, holds] for key, holds in solution.predicates.items()],
        }
        stored.append(record)
        self.stats["fixpoints added"] += 1
        self._append(record)

    def fixpoint(
        self: Cache, instance: str, mutations: dict[str, int], reactions: set[str]
    ) -> Solution | None:
        # > A fixpoint stored under mutations A holds under B if B keeps A, each
        # > mutation of B forces its node to its value in the fixpoint, and a
        # > reaction forced to 0 carries no flux (it may have had no node under A)
        mutations = {node: int(value) for node, value in mutations.items()}
        for record in self.fixpoints.get(instance, []):
            ones = set(record["ones"])
            fluxes = record["fluxes"]
            kept = all(mutations.get(node) == value for node, value in record["mutations"].items())
            same = all((node in ones) == bool(value) for node, value in mutations.items())
            unused = all(
                abs(fluxes.get(node, 0.0)) <= UNUSED
                for node, value in mutations.items() if node in reactions and not value
            )
            if kept and same and unused:
                self.stats["fixpoints reused"] += 1
                return self._solution(record)
        return None

    def neighbour(self: Cache, instance: str) -> Solution | None:
        # ~ Any fixpoint stored for the instance: a starting point for a mutant
        stored = self.fixpoints.get(instance)
        return self._solution(stored[0]) if stored else None

    @staticmethod
    def _solution(record: dict) -> Solution:
        predicates = {(v, op, float(b)): bool(holds) for v, op, b, holds in record["predicates"]}
        return Solution(set(record["ones"]), dict(record["fluxes"]), predicates)
