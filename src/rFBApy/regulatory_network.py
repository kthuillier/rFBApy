# ==============================================================================
# Imports
# ==============================================================================
from __future__ import annotations

import os
import re
import xml.etree.ElementTree as ET
from collections.abc import Callable

from networkx import MultiDiGraph

from rFBApy.asp import asp_number, asp_string, write_asp
from rFBApy.bn.bexpr import (
    FALSE,
    TRUE,
    AndExpr,
    BoolExpr,
    FalseExpr,
    NotExpr,
    OrExpr,
    ThresholdExpr,
    TrueExpr,
    VarExpr,
    parse,
)
from rFBApy.sbml import attr, local, read_model, sections

# ==============================================================================
# SBML-Qual parsing helpers
# ==============================================================================
# > Qualitative species carry an (informal but consistent) <notes> annotation of
# > the form "STATE <level>:<interval>" mapping each discrete qual:level to the
# > interval of the underlying continuous quantity (reaction flux / metabolite
# > concentration) it stands for, e.g. "STATE 0:[0,0]" / "STATE 1:]0,+inf]".
# > Species without such notes are plain boolean regulatory nodes, where the
# > level *is* the value.
# ==============================================================================
_STATE_NOTE_RE = re.compile(r"STATE\s*(\d+)\s*:\s*([^\s<]+)")
_INTERVAL_RE = re.compile(
    r"^([\[\]])\s*(-?inf|[+-]?\d+(?:\.\d+)?)\s*,\s*"
    r"(\+?inf|[+-]?\d+(?:\.\d+)?)\s*([\[\]])$"
)

_MATHML_COMPARISON_OPS: dict[str, str] = {
    "eq": "==",
    "neq": "!=",
    "geq": ">=",
    "leq": "<=",
    "gt": ">",
    "lt": "<",
}

# > Bounds of a STATE interval: (lo, lo_open, hi, hi_open); `None` means "ND"
# > (not defined), i.e. a level that is never meant to be tested against.
_Interval = tuple[float, bool, float, bool] | None


def _parse_bound(token: str) -> float:
    token = token.strip()
    if token in ("+inf", "inf"):
        return float("inf")
    if token == "-inf":
        return float("-inf")
    return float(token)


def _parse_state_intervals(notes: str) -> dict[int, _Interval]:
    intervals: dict[int, _Interval] = {}
    for level_str, spec in _STATE_NOTE_RE.findall(notes):
        level: int = int(level_str)
        spec = spec.strip()
        if spec.upper() == "ND":
            intervals[level] = None
            continue
        m = _INTERVAL_RE.match(spec)
        if m is None:
            raise ValueError(f"Cannot parse SBML-Qual STATE interval: {spec!r}")
        lo_bracket, lo, hi, hi_bracket = m.groups()
        intervals[level] = (
            _parse_bound(lo),
            lo_bracket == "]",
            _parse_bound(hi),
            hi_bracket == "[",
        )
    return intervals


def _interval_of(
    species: str, level: int, intervals: dict[int, _Interval]
) -> tuple[float, bool, float, bool]:
    if level not in intervals:
        raise ValueError(
            f"Qualitative species '{species}' has no STATE definition for level {level}"
        )
    interval: _Interval = intervals[level]
    if interval is None:
        raise NotImplementedError(
            f"Cannot translate reference to the undefined ('ND') state "
            f"{level} of qualitative species '{species}'"
        )
    return interval


def _range_expr(
    species: str, lo: float, lo_open: bool, hi: float, hi_open: bool
) -> BoolExpr:
    if lo == hi:
        return ThresholdExpr(VarExpr(species), "==", lo)
    parts: list[BoolExpr] = []
    if lo != float("-inf"):
        parts.append(ThresholdExpr(VarExpr(species), ">" if lo_open else ">=", lo))
    if hi != float("inf"):
        parts.append(ThresholdExpr(VarExpr(species), "<" if hi_open else "<=", hi))
    if len(parts) == 0:
        return TRUE
    if len(parts) == 1:
        return parts[0]
    return AndExpr(parts)


def _level_eq_expr(
    species: str, level: int, intervals: dict[int, _Interval]
) -> BoolExpr:
    lo, lo_open, hi, hi_open = _interval_of(species, level, intervals)
    return _range_expr(species, lo, lo_open, hi, hi_open)


def _level_in_expr(
    species: str, levels: list[int], intervals: dict[int, _Interval]
) -> BoolExpr:
    # > Union of the (contiguous, ordered) STATE intervals of several levels of
    # > the same species, e.g. "X == 1 | X == 2" over adjacent levels 1/2 is
    # > exactly the merged interval spanning both, and collapses to a single
    # > threshold whenever that merged interval is one-sided (e.g. "X > 0").
    sorted_levels: list[int] = sorted(set(levels))
    runs: list[tuple[int, int]] = []
    start = prev = sorted_levels[0]
    for level in sorted_levels[1:]:
        if level == prev + 1:
            prev = level
            continue
        runs.append((start, prev))
        start = prev = level
    runs.append((start, prev))

    parts: list[BoolExpr] = []
    for lo_level, hi_level in runs:
        lo, lo_open, _, _ = _interval_of(species, lo_level, intervals)
        _, _, hi, hi_open = _interval_of(species, hi_level, intervals)
        parts.append(_range_expr(species, lo, lo_open, hi, hi_open))
    return parts[0] if len(parts) == 1 else OrExpr(parts)


def _level_geq_expr(
    species: str, level: int, intervals: dict[int, _Interval]
) -> BoolExpr:
    lo, lo_open, _, _ = _interval_of(species, level, intervals)
    if lo == float("-inf"):
        return TRUE
    return ThresholdExpr(VarExpr(species), ">" if lo_open else ">=", lo)


def _level_leq_expr(
    species: str, level: int, intervals: dict[int, _Interval]
) -> BoolExpr:
    _, _, hi, hi_open = _interval_of(species, level, intervals)
    if hi == float("inf"):
        return TRUE
    return ThresholdExpr(VarExpr(species), "<" if hi_open else "<=", hi)


def _level_expr(
    species: str,
    op: str,
    level: int,
    species_intervals: dict[str, dict[int, _Interval] | None],
) -> BoolExpr:
    intervals: dict[int, _Interval] | None = species_intervals.get(species)
    # ~ Plain boolean species (no STATE notes): the level *is* the raw value
    if intervals is None:
        if op == "==":
            if level == 1:
                return VarExpr(species)
            if level == 0:
                return NotExpr(VarExpr(species))
            return ThresholdExpr(VarExpr(species), "==", float(level))
        if op == "!=":
            return NotExpr(_level_expr(species, "==", level, species_intervals))
        return ThresholdExpr(VarExpr(species), op, float(level))
    # ~ Thresholded species (reaction flux / metabolite concentration)
    if op == "==":
        return _level_eq_expr(species, level, intervals)
    if op == "!=":
        return NotExpr(_level_eq_expr(species, level, intervals))
    if op == ">=":
        return _level_geq_expr(species, level, intervals)
    if op == "<=":
        return _level_leq_expr(species, level, intervals)
    if op == ">":
        return NotExpr(_level_leq_expr(species, level, intervals))
    if op == "<":
        return NotExpr(_level_geq_expr(species, level, intervals))
    raise NotImplementedError(f"Unsupported SBML-Qual comparison operator: {op}")


def _unwrap(node: ET.Element) -> ET.Element:
    # ~ <apply><apply>...</apply></apply> is the inner <apply>
    while local(node.tag) == "apply" and len(node) == 1 and local(node[0].tag) == "apply":
        node = node[0]
    return node


def _op(node: ET.Element) -> str:
    # ~ MathML: <apply><and/>...</apply> is the operator "and"; a leaf (<ci>,
    # <cn>, <true/>) is its own tag
    node = _unwrap(node)
    return local(node[0].tag) if local(node.tag) == "apply" else local(node.tag)


def _args(node: ET.Element) -> list[ET.Element]:
    node = _unwrap(node)
    return list(node)[1:] if local(node.tag) == "apply" else []


def _number(node: ET.Element) -> float | None:
    if local(node.tag) != "cn":
        return None
    try:
        return float("".join(node.itertext()).strip())
    except ValueError:
        return None


def _name(node: ET.Element) -> str:
    return "".join(node.itertext()).strip()


def _name_and_level(left: ET.Element, right: ET.Element) -> tuple[str, int] | None:
    value: float | None
    if local(left.tag) == "ci" and (value := _number(right)) is not None:
        return _name(left), round(value)
    if local(right.tag) == "ci" and (value := _number(left)) is not None:
        return _name(right), round(value)
    return None


def _flatten_or(node: ET.Element) -> list[ET.Element]:
    if _op(node) != "or":
        return [node]
    children: list[ET.Element] = []
    for child in _args(node):
        children.extend(_flatten_or(child))
    return children


def _or_expr(
    node: ET.Element,
    species_intervals: dict[str, dict[int, _Interval] | None],
) -> BoolExpr:
    # > Flatten nested "or"s and merge equality checks against the same
    # > thresholded species into a single (possibly simplified) range, e.g.
    # > "X == 1 | X == 2 | Y == 1" -> "[X > 0] | [Y > 0]" instead of the
    # > redundant "(X > 0 & X <= 10 | X > 10) | Y > 0".
    levels_by_species: dict[str, list[int]] = {}
    other: list[BoolExpr] = []
    for child in _flatten_or(node):
        name_level = (
            _name_and_level(*_args(child))
            if _op(child) == "eq" and len(_args(child)) == 2
            else None
        )
        if name_level is not None and species_intervals.get(name_level[0]) is not None:
            species, level = name_level
            levels_by_species.setdefault(species, []).append(level)
        else:
            other.append(_ast_to_expr(child, species_intervals))
    parts: list[BoolExpr] = other + [
        _level_in_expr(species, levels, species_intervals[species]) # type: ignore
        for species, levels in levels_by_species.items()
    ]
    return parts[0] if len(parts) == 1 else OrExpr(parts)


def _ast_to_expr(
    node: ET.Element,
    species_intervals: dict[str, dict[int, _Interval] | None],
) -> BoolExpr:
    op: str = _op(node)
    args: list[ET.Element] = _args(node)
    if op == "and":
        return AndExpr([_ast_to_expr(a, species_intervals) for a in args])
    if op == "or":
        return _or_expr(node, species_intervals)
    if op == "not":
        return NotExpr(_ast_to_expr(args[0], species_intervals))
    if op == "true":
        return TRUE
    if op == "false":
        return FALSE
    if op == "ci":
        return _level_expr(_name(node), "==", 1, species_intervals)
    if op in _MATHML_COMPARISON_OPS:
        if len(args) != 2:
            raise NotImplementedError(
                "SBML-Qual comparison node without exactly 2 children"
            )
        name_level = _name_and_level(args[0], args[1])
        if name_level is None:
            raise NotImplementedError(
                "SBML-Qual comparisons must be between a qualitative species and "
                "an integer level"
            )
        species, level = name_level
        return _level_expr(
            species, _MATHML_COMPARISON_OPS[op], level, species_intervals
        )
    raise NotImplementedError(f"Unsupported SBML-Qual MathML node type: {op}")


def _transition_expr(
    transition: ET.Element,
    species_intervals: dict[str, dict[int, _Interval] | None],
) -> BoolExpr:
    tid: str | None = attr(transition, "id")
    default_level: int = 0
    terms_1: list[ET.Element] = []
    terms_0: list[ET.Element] = []
    for part in sections(transition).get("listOfFunctionTerms", ()):
        level: int = int(attr(part, "resultLevel") or 0)
        if local(part.tag) == "defaultTerm":
            default_level = level
        elif local(part.tag) == "functionTerm":
            math: ET.Element = next(e for e in part if local(e.tag) == "math")
            (terms_1 if level == 1 else terms_0).append(math[0])

    if len(terms_1) > 0 and len(terms_0) == 0:
        if default_level != 0:
            raise NotImplementedError(
                f"Transition '{tid}' has resultLevel-1 function "
                f"terms but a non-zero default level ({default_level})"
            )
        exprs = [_ast_to_expr(m, species_intervals) for m in terms_1]
        return exprs[0] if len(exprs) == 1 else OrExpr(exprs)
    if len(terms_0) > 0 and len(terms_1) == 0:
        if default_level != 1:
            raise NotImplementedError(
                f"Transition '{tid}' has resultLevel-0 function "
                f"terms but a non-one default level ({default_level})"
            )
        exprs = [_ast_to_expr(m, species_intervals) for m in terms_0]
        return NotExpr(exprs[0] if len(exprs) == 1 else OrExpr(exprs))
    if len(terms_1) == 0 and len(terms_0) == 0:
        return TRUE if default_level == 1 else FALSE
    raise NotImplementedError(
        f"Transition '{tid}' mixes resultLevel-0 and resultLevel-1 "
        "function terms, which is not supported"
    )


# ==============================================================================
# SBML-Qual writing helpers
# ==============================================================================
# > The inverse of the parsing helpers above: build, for every variable that is
# > ever compared to a threshold, the *finest* STATE-interval partition induced
# > by its distinct comparison values -- (-inf, v0), [v0,v0], (v0,v1), [v1,v1],
# > ... -- so that any of "==", "!=", "<", "<=", ">", ">=" against any of those
# > values can be written as the *same* relational MathML operator against the
# > (unique) integer level of that value's singleton interval, and read back
# > through `_level_expr` bit-for-bit unchanged.
# ==============================================================================
_MATHML_OP: dict[str, str] = {
    "==": "eq", "!=": "neq", ">=": "geq", "<=": "leq", ">": "gt", "<": "lt",
}


def _canonical_partition(
    values: set[float],
) -> tuple[dict[int, _Interval], dict[float, int]]:
    intervals: dict[int, _Interval] = {}
    value_to_level: dict[float, int] = {}
    level = 0
    prev = float("-inf")
    for v in sorted(values):
        intervals[level] = (prev, True, v, True)
        level += 1
        intervals[level] = (v, False, v, False)
        value_to_level[v] = level
        level += 1
        prev = v
    intervals[level] = (prev, True, float("inf"), False)
    return intervals, value_to_level


def _fmt_bound(v: float) -> str:
    if v == float("inf"):
        return "+inf"
    if v == float("-inf"):
        return "-inf"
    if v == int(v):
        return str(int(v))
    # ~ Fixed-point notation only: the STATE-interval grammar (mirroring the
    # ~ original models' own convention, e.g. "0.000000004") has no support
    # ~ for scientific notation, unlike Python's `str()` on small floats.
    return f"{v:.15f}".rstrip("0").rstrip(".")


def _species_notes(intervals: dict[int, _Interval]) -> str:
    lines = "".join(
        f"<p>STATE {level}:{']' if lo_open else '['}{_fmt_bound(lo)},"
        f"{_fmt_bound(hi)}{'[' if hi_open else ']'}</p>"
        for level, (lo, lo_open, hi, hi_open) in sorted(intervals.items()) # type: ignore
    )
    return (
        '<notes><body xmlns="http://www.w3.org/1999/xhtml">'
        f"{lines}</body></notes>"
    )


def _sign_str(sign: int) -> str:
    return {1: "positive", -1: "negative"}.get(sign, "unknown")


def _expr_to_mathml(
    expr: BoolExpr, value_to_level: dict[str, dict[float, int]]
) -> str:
    if isinstance(expr, TrueExpr):
        return "<true/>"
    if isinstance(expr, FalseExpr):
        return "<false/>"
    if isinstance(expr, NotExpr):
        return f"<apply><not/>{_expr_to_mathml(expr.item, value_to_level)}</apply>"
    if isinstance(expr, AndExpr):
        children = "".join(_expr_to_mathml(x, value_to_level) for x in expr.items)
        return f"<apply><and/>{children}</apply>"
    if isinstance(expr, OrExpr):
        children = "".join(_expr_to_mathml(x, value_to_level) for x in expr.items)
        return f"<apply><or/>{children}</apply>"
    if isinstance(expr, ThresholdExpr):
        name = expr.variable.name
        level = value_to_level[name][expr.value]
        return (
            f"<apply><{_MATHML_OP[expr.op]}/><ci> {name} </ci>"
            f'<cn type="integer"> {level} </cn></apply>'
        )
    if isinstance(expr, VarExpr):
        name = expr.name
        if name in value_to_level:
            # ~ Thresholded species also referenced as a bare boolean:
            # ~ "truthy" <=> "!= 0"
            level = value_to_level[name][0.0]
            return (
                f"<apply><neq/><ci> {name} </ci>"
                f'<cn type="integer"> {level} </cn></apply>'
            )
        return '<apply><eq/><ci> ' + name + ' </ci><cn type="integer"> 1 </cn></apply>'
    raise NotImplementedError(f"Cannot translate {type(expr).__name__} to MathML")


# ==============================================================================
# ASP export helpers
# ==============================================================================
# > fixpoint.lp only knows [A > b] (gt) and [A < b] (lt): each operator is a
# > DNF over them, as (predicate, sign).
# ==============================================================================
_ASP_COMPARISON_OPS: dict[str, list[list[tuple[str, int]]]] = {
    ">": [[("gt", 1)]],
    "<": [[("lt", 1)]],
    "<=": [[("gt", -1)]],
    ">=": [[("lt", -1)]],
    "==": [[("gt", -1), ("lt", -1)]],
    "!=": [[("gt", 1)], [("lt", 1)]],
}


def _asp_canonical_thresholds(
    expr: BoolExpr,
    predicates: dict[tuple[str, float, str], VarExpr],
) -> BoolExpr:
    # ~ Each gt/lt predicate becomes one placeholder variable, so that the DNF
    # ~ simplifies [A <= 0] against [A > 0]
    if isinstance(expr, (AndExpr, OrExpr)):
        return type(expr)(
            [_asp_canonical_thresholds(item, predicates) for item in expr.items]
        )
    if isinstance(expr, NotExpr):
        return NotExpr(_asp_canonical_thresholds(expr.item, predicates))
    if not isinstance(expr, ThresholdExpr):
        return expr
    conjunctions: list[BoolExpr] = []
    for conjunction in _ASP_COMPARISON_OPS[expr.op]:
        literals: list[BoolExpr] = []
        for op, sign in conjunction:
            key: tuple[str, float, str] = (expr.variable.name, float(expr.value), op)
            # ~ "\0" cannot start a parsed name: no clash with a real variable
            atom: VarExpr = predicates.setdefault(key, VarExpr(f"\0{len(predicates)}"))
            literals.append(atom if sign > 0 else NotExpr(atom))
        conjunctions.append(literals[0] if len(literals) == 1 else AndExpr(literals))
    return conjunctions[0] if len(conjunctions) == 1 else OrExpr(conjunctions)


def _asp_dnf(expr: BoolExpr) -> frozenset[frozenset[tuple[str, int]]]:
    # ~ Clauses as sets of (name, sign). No clause: false; an empty clause: true
    clauses: set[frozenset[tuple[str, int]]] = set()
    for term in expr.to_dnf().items:
        if term is FALSE:
            continue
        literals: list[BoolExpr]
        if term is TRUE:
            literals = []
        elif isinstance(term, AndExpr):
            literals = term.items
        else:
            literals = [term]
        clauses.add(
            frozenset(
                (lit.item.name, -1) if isinstance(lit, NotExpr) else (lit.name, 1)  # type: ignore
                for lit in literals
            )
        )
    return frozenset(clauses)


# ==============================================================================
# Regulatory Network
# ==============================================================================
class RegulatoryNetwork:
    # ==========================================================================
    # Initialization
    # ==========================================================================
    def __init__(self: RegulatoryNetwork):
        self.__rules: dict[str, BoolExpr] = {}
        self.__fn: dict[
            str, Callable[[dict[str, bool | int | float]], bool | None]
        ] = {}

        self.__components: frozenset[str] | None = None
        self.__constants: frozenset[str] | None = None
        self.__thresholds: frozenset[tuple[str, float]] | None = None

    # ==========================================================================
    # Properties
    # ==========================================================================
    @property
    def nodes(self: RegulatoryNetwork) -> frozenset[str]:
        return frozenset(self.__rules)

    @property
    def components(self: RegulatoryNetwork) -> frozenset[str]:
        if self.__components is None:
            self.__components = frozenset(self.__rules.keys()).union(
                *(rule.variables for rule in self.__rules.values())
            )
        return self.__components

    @property
    def influence_graph(self: RegulatoryNetwork) -> MultiDiGraph:
        g: MultiDiGraph = MultiDiGraph()
        for n, rule in self.__rules.items():
            signed_vars: frozenset[tuple[str, int]] = rule.signed_variables
            for u, s in signed_vars:
                g.add_edge(u, n, sign=s)
        return g

    @property
    def undefined(self: RegulatoryNetwork) -> frozenset[str]:
        return self.components.difference(self.nodes)

    @property
    def constants(self: RegulatoryNetwork) -> frozenset[str]:
        if self.__constants is None:
            self.__constants = frozenset(
                [n for n, rule in self.__rules.items() if rule.is_constant()]
            )
        return self.__constants

    @property
    def thresholds(self: RegulatoryNetwork) -> frozenset[tuple[str, float]]:
        if self.__thresholds is None:
            self.__thresholds = frozenset().union(
                *(rule.thresholds for rule in self.__rules.values())
            )
        return self.__thresholds

    # ==========================================================================
    # Getters / Setters
    # ==========================================================================
    def __iter__(self: RegulatoryNetwork):
        yield from self.__rules

    def __contains__(self: RegulatoryNetwork, node: object) -> bool:
        # ~ Without it, `node in bn` scans the generator of __iter__: O(nodes)
        return node in self.__rules

    def __getitem__(self: RegulatoryNetwork, node: str) -> BoolExpr:
        return self.__rules[node]

    def __setitem__(
        self: RegulatoryNetwork,
        node: str,
        rule: str | int | BoolExpr,
    ) -> None:
        if isinstance(rule, int):
            rule = str(rule)
        self.__rules[node] = rule if isinstance(rule, BoolExpr) else parse(rule)
        self.__fn[node] = self.__rules[node].simplify().compile()
        # ~ Reset all memoized elements
        self.__thresholds = None
        self.__constants = None
        self.__components = None

    def __delitem__(self: RegulatoryNetwork, node: str):
        del self.__rules[node]
        del self.__fn[node]
        self.__thresholds = None
        self.__constants = None
        self.__components = None

    # ==========================================================================
    # Evaluation
    # ==========================================================================
    def __call__(
        self: RegulatoryNetwork,
        context: dict[str, int | float | bool],
    ) -> dict[str, int | None]:
        vals: dict[str, bool | None] = {n: fn(context) for n, fn in self.__fn.items()}
        return {n: None if v is None else (1 if v else 0) for n, v in vals.items()}

    # ==========================================================================
    # Auxiliary functions
    # ==========================================================================
    def copy(self: RegulatoryNetwork) -> RegulatoryNetwork:
        rn: RegulatoryNetwork = RegulatoryNetwork()
        rn.__rules = self.__rules.copy()
        rn.__fn = self.__fn.copy()
        return rn

    # ==========================================================================
    # Display
    # ==========================================================================
    def __str__(self: RegulatoryNetwork) -> str:
        return "\n".join(f"{n} <- {rule}" for n, rule in self.__rules.items())

    def __repr__(self: RegulatoryNetwork) -> str:
        return (
            f"{type(self).__name__}("
            + ", ".join(f"Rule({n};{rule!r}" for n, rule in self.__rules.items())
            + ")"
        )

    # ==========================================================================
    # Parsing
    # ==========================================================================
    # --------------------------------------------------------------------------
    # BNET
    # --------------------------------------------------------------------------
    @staticmethod
    def load_bnet(bnet: str) -> RegulatoryNetwork:
        if not os.path.isfile(bnet):
            raise FileNotFoundError(f"Regulatory network BNET file not found: {bnet}")
        # ~ Initialize output --------------------------------------------------
        rn = RegulatoryNetwork()
        # ~ Read BNET file -----------------------------------------------------
        with open(bnet, "r", encoding="utf-8") as file:
            for line in file.readlines():  # noqa: FURB129
                line: str = line.split("#")[0].strip()  # remove comments
                if len(line) == 0:
                    continue
                sep: str = "->" if "->" in line else ","
                node, rule_str = [e.strip() for e in line.split(sep)[:2]]
                rn[node] = rule_str
        # ~ Output -------------------------------------------------------------
        return rn

    # --------------------------------------------------------------------------
    # SBML-Qual
    # --------------------------------------------------------------------------
    @staticmethod
    def load_sbml(sbml: str) -> RegulatoryNetwork:
        # ~ Read SBML-Qual file --------------------------------------------------
        model: ET.Element = read_model(sbml)
        lists: dict[str, ET.Element] = sections(model)
        if "listOfQualitativeSpecies" not in lists and "listOfTransitions" not in lists:
            raise ValueError(f"Not an SBML-Qual (regulatory network) file: {sbml}")
        # ~ Species STATE intervals (thresholded species) / booleans (no notes) --
        species_intervals: dict[str, dict[int, _Interval] | None] = {}
        for sp in lists.get("listOfQualitativeSpecies", ()):
            notes: ET.Element | None = next(
                (e for e in sp if local(e.tag) == "notes"), None
            )
            species_intervals[str(attr(sp, "id"))] = (
                _parse_state_intervals(ET.tostring(notes, encoding="unicode"))
                if notes is not None
                else None
            )
        # ~ Initialize output ------------------------------------------------------
        rn = RegulatoryNetwork()
        # ~ Translate each transition into a Boolean rule -------------------------
        for transition in lists.get("listOfTransitions", ()):
            outputs: list[ET.Element] = list(
                sections(transition).get("listOfOutputs", ())
            )
            if len(outputs) != 1:
                raise NotImplementedError(
                    f"Transition '{attr(transition, 'id')}' does not have exactly "
                    "one output"
                )
            node: str = str(attr(outputs[0], "qualitativeSpecies"))
            rn[node] = _transition_expr(transition, species_intervals)
        # ~ Output -------------------------------------------------------------
        return rn

    # ==========================================================================
    # Export
    # ==========================================================================
    def to_bnet(self: RegulatoryNetwork, filename: str) -> None:
        lines = [f"{n} -> {rule!s}\n" for n, rule in self.__rules.items()]
        with open(filename, "w", encoding="utf-8") as file:
            file.writelines(lines)

    def to_sbmlqual(self: RegulatoryNetwork, filename: str) -> None:
        # ~ Per-variable threshold values (network-wide, so every rule that
        # ~ references a given species agrees on the same level partition) --
        threshold_values: dict[str, set[float]] = {}
        for var, val in self.thresholds:
            threshold_values.setdefault(var, set()).add(val)

        # ~ Species also referenced as a bare boolean need level 0.0 too, so
        # ~ that usage can be written as "!= 0" against the same partition ---
        bool_usage: set[str] = set()

        def _collect_bool_usage(expr: BoolExpr) -> None:
            if isinstance(expr, VarExpr):
                bool_usage.add(expr.name)
            elif isinstance(expr, NotExpr):
                _collect_bool_usage(expr.item)
            elif isinstance(expr, (AndExpr, OrExpr)):
                for item in expr.items:
                    _collect_bool_usage(item)

        for rule in self.__rules.values():
            _collect_bool_usage(rule)
        for name in bool_usage & threshold_values.keys():
            threshold_values[name].add(0.0)

        partitions: dict[str, dict[int, _Interval]] = {}
        value_to_level: dict[str, dict[float, int]] = {}
        for var, vals in threshold_values.items():
            partitions[var], value_to_level[var] = _canonical_partition(vals)

        # ~ Qualitative species --------------------------------------------------
        species_xml: list[str] = []
        for name in sorted(self.components):
            if name in partitions:
                species_xml.append(
                    f'<qual:qualitativeSpecies qual:compartment="default" '
                    f'qual:constant="false" qual:id="{name}" '
                    f'qual:maxLevel="{max(partitions[name])}" '
                    f'qual:initialLevel="0">{_species_notes(partitions[name])}'
                    "</qual:qualitativeSpecies>"
                )
            else:
                species_xml.append(
                    f'<qual:qualitativeSpecies qual:compartment="default" '
                    f'qual:constant="false" qual:id="{name}" qual:maxLevel="1" '
                    f'qual:initialLevel="0"/>'
                )

        # ~ Transitions ------------------------------------------------------------
        transitions_xml: list[str] = []
        for node in sorted(self.nodes):
            rule: BoolExpr = self.__rules[node]
            inputs = "".join(
                f'<qual:input qual:id="tr_{node}_in_{var}" '
                f'qual:qualitativeSpecies="{var}" qual:sign="{_sign_str(sign)}" '
                'qual:transitionEffect="none"/>'
                for var, sign in sorted(rule.signed_variables)
            )
            list_of_inputs = f"<qual:listOfInputs>{inputs}</qual:listOfInputs>" if inputs else ""
            output = (
                f'<qual:output qual:id="tr_{node}_out" qual:qualitativeSpecies="{node}" '
                'qual:transitionEffect="assignmentLevel"/>'
            )
            simplified: BoolExpr = rule.simplify()
            if isinstance(simplified, (TrueExpr, FalseExpr)):
                default_level: int = 1 if isinstance(simplified, TrueExpr) else 0
                function_terms = f'<qual:defaultTerm qual:resultLevel="{default_level}"/>'
            else:
                math = _expr_to_mathml(rule, value_to_level)
                function_terms = (
                    '<qual:defaultTerm qual:resultLevel="0"/>'
                    '<qual:functionTerm qual:resultLevel="1">'
                    f'<math xmlns="http://www.w3.org/1998/Math/MathML">{math}</math>'
                    "</qual:functionTerm>"
                )
            transitions_xml.append(
                f'<qual:transition qual:id="tr_{node}_">{list_of_inputs}'
                f"<qual:listOfOutputs>{output}</qual:listOfOutputs>"
                f"<qual:listOfFunctionTerms>{function_terms}</qual:listOfFunctionTerms>"
                "</qual:transition>"
            )

        doc = (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" '
            'xmlns:qual="http://www.sbml.org/sbml/level3/version1/qual/version1" '
            'level="3" version="1" qual:required="true">\n'
            '<model id="rFBApy_regulatory_network">\n'
            "<listOfCompartments>\n"
            '<compartment id="default" constant="true"/>\n'
            "</listOfCompartments>\n"
            "<qual:listOfQualitativeSpecies>\n"
            + "\n".join(species_xml) + "\n"
            "</qual:listOfQualitativeSpecies>\n"
            "<qual:listOfTransitions>\n"
            + "\n".join(transitions_xml) + "\n"
            "</qual:listOfTransitions>\n"
            "</model>\n"
            "</sbml>\n"
        )
        with open(filename, "w", encoding="utf-8") as file:
            file.write(doc)

    def to_asp(self: RegulatoryNetwork, filename: str | None = None) -> str:
        # ~ cst/2, rule/2 and clause/1,3 facts of fixpoint/asp/fixpoint.lp, rules
        # ~ in DNF; identical clauses share one id across nodes
        predicates: dict[tuple[str, float, str], VarExpr] = {}
        dnfs: dict[str, frozenset[frozenset[tuple[str, int]]]] = {
            n: _asp_dnf(_asp_canonical_thresholds(rule, predicates))
            for n, rule in self.__rules.items()
        }
        # ~ Placeholder -> thr/3 term (flux or concentration: decided in ASP)
        terms: dict[str, str] = {
            placeholder.name: f"thr({asp_string(var)},{op},{asp_number(val)})"
            for (var, val, op), placeholder in predicates.items()
        }

        lines: list[str] = [
            f"% Boolean network: {len(dnfs)} nodes, {len(predicates)} threshold predicates"
        ]
        clause_ids: dict[frozenset[tuple[str, int]], int] = {}
        for n, dnf in dnfs.items():
            if len(dnf) == 0:
                lines.append(f"cst({asp_string(n)},-1).")
                continue
            if frozenset() in dnf:
                lines.append(f"cst({asp_string(n)},1).")
                continue
            for clause in sorted(dnf, key=sorted):
                if clause not in clause_ids:
                    clause_ids[clause] = len(clause_ids)
                    lines.append(f"clause({clause_ids[clause]}).")
                    for var, sign in sorted(clause):
                        literal: str = terms.get(var, asp_string(var))
                        lines.append(f"clause({clause_ids[clause]},{literal},{sign}).")
                lines.append(f"rule({asp_string(n)},{clause_ids[clause]}).")
        return write_asp(lines, filename)


# ==============================================================================
# Main
# ==============================================================================
if __name__ == "__main__":
    ...