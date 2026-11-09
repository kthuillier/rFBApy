class Fba:
    """Exact flux balance analysis, plain and parsimonious."""

    def __init__(
        self,
        rows: int,
        stoichiometry: list[tuple[int, int, float]],
        bounds: list[tuple[float, float]],
        objective: int,
    ) -> None: ...
    def solve(
        self, bounds: list[tuple[int, float, float]], pfba: bool
    ) -> tuple[float | None, list[float]]: ...
