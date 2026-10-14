# ==============================================================================
# Imports
# ==============================================================================
from __future__ import annotations

from merrinasp.theory.lra.models.interface import ModelInterface

# ==============================================================================
# LP conflicts by bisection
# ==============================================================================
# > merrinasp's ModelInterface.core_unsat_exists removes the propagation groups
# > (`added_order`) one by one, first to last, putting a group back when the LP
# > becomes feasible: one LP solve per group. Its answer is the latest group G
# > such that G and the groups after it are infeasible. Removing groups from
# > the front only turns infeasible into feasible, so G is found by bisection,
# > then the search goes on after G until the kept groups are infeasible.
# > Same conflict, about log2(#groups) solves. Belongs in merrinasp.
# ==============================================================================
_ORIGINAL = ModelInterface.core_unsat_exists


def core_unsat_exists(self: ModelInterface) -> list[int]:
    groups = [
        [cid for cid in batch if cid in self.constraints_exists] for batch in self.added_order
    ]
    count = len(groups)
    active = [True] * count
    out: dict[int, int] = {}  # removed constraint -> its description

    def switch(index: int, on: bool) -> None:
        for cid in groups[index]:
            if on:
                self.constraints[cid] = self._add_lpconstraint(cid)
                self.description[cid] = out.pop(cid)
            else:
                self._remove_lpconstraint(self.constraints[cid])
                out[cid] = self.description[cid]
                del self.constraints[cid]
                del self.description[cid]
        active[index] = on

    def infeasible(kept: set[int], start: int) -> bool:
        # ~ Groups of `kept` and those from `start` on
        for index in range(count):
            wanted = index in kept or index >= start
            if wanted != active[index]:
                switch(index, wanted)
        return not self.check_exists()

    # ~ Group 0 (decision level 0: steady state, bounds, medium) is always kept
    kept = {0} if count else set()
    first = 1
    # ~ Invariant: `kept` and the groups from `first` on are infeasible
    while not infeasible(kept, count):
        if first >= count:
            # ~ The solver contradicts its own verdict at the edge of its
            # ~ tolerance: the whole program is the conflict
            kept = set(range(count))
            break
        low, high = first, count - 1
        while low < high:
            middle = (low + high + 1) // 2
            if infeasible(kept, middle):
                low = middle
            else:
                high = middle - 1
        kept.add(low)
        first = low + 1
    for index in range(count):
        if not active[index]:
            switch(index, True)
    self.logger.conflicts_exists += 1
    return [abs(cid) for index in sorted(kept) for cid in groups[index]]


def install() -> None:
    ModelInterface.core_unsat_exists = core_unsat_exists  # type: ignore


def uninstall() -> None:
    ModelInterface.core_unsat_exists = _ORIGINAL  # type: ignore
