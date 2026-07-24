# ==============================================================================
# Imports
# ==============================================================================
from __future__ import annotations

import os
import re
from collections.abc import Callable

from libsbml import (  # type: ignore
    AST_CONSTANT_FALSE,
    AST_CONSTANT_TRUE,
    AST_LOGICAL_AND,
    AST_LOGICAL_NOT,
    AST_LOGICAL_OR,
    AST_NAME,
    AST_RELATIONAL_EQ,
    AST_RELATIONAL_GEQ,
    AST_RELATIONAL_GT,
    AST_RELATIONAL_LEQ,
    AST_RELATIONAL_LT,
    AST_RELATIONAL_NEQ,
    ASTNode,
    Model,
    QualitativeSpecies,
    QualModelPlugin,
    SBMLDocument,
    SBMLReader,
    Transition,
)
from networkx import MultiDiGraph

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

_AST_COMPARISON_OPS: dict[int, str] = {
    AST_RELATIONAL_EQ: "==",
    AST_RELATIONAL_NEQ: "!=",
    AST_RELATIONAL_GEQ: ">=",
    AST_RELATIONAL_LEQ: "<=",
    AST_RELATIONAL_GT: ">",
    AST_RELATIONAL_LT: "<",
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


def _name_and_level(left: ASTNode, right: ASTNode) -> tuple[str, int] | None:
    if left.isName() and (right.isInteger() or right.isReal()):
        return left.getName(), round(right.getValue())
    if right.isName() and (left.isInteger() or left.isReal()):
        return right.getName(), round(left.getValue())
    return None


def _flatten_or(node: ASTNode) -> list[ASTNode]:
    if node.getType() != AST_LOGICAL_OR:
        return [node]
    children: list[ASTNode] = []
    for i in range(node.getNumChildren()):
        children.extend(_flatten_or(node.getChild(i)))
    return children


def _or_expr(
    node: ASTNode,
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
            _name_and_level(child.getChild(0), child.getChild(1))
            if child.getType() == AST_RELATIONAL_EQ and child.getNumChildren() == 2
            else None
        )
        if name_level is not None and species_intervals.get(name_level[0]) is not None:
            species, level = name_level
            levels_by_species.setdefault(species, []).append(level)
        else:
            other.append(_ast_to_expr(child, species_intervals))
    parts: list[BoolExpr] = other + [
        _level_in_expr(species, levels, species_intervals[species])
        for species, levels in levels_by_species.items()
    ]
    return parts[0] if len(parts) == 1 else OrExpr(parts)


def _ast_to_expr(
    node: ASTNode,
    species_intervals: dict[str, dict[int, _Interval] | None],
) -> BoolExpr:
    node_type: int = node.getType()
    if node_type == AST_LOGICAL_AND:
        return AndExpr(
            [
                _ast_to_expr(node.getChild(i), species_intervals)
                for i in range(node.getNumChildren())
            ]
        )
    if node_type == AST_LOGICAL_OR:
        return _or_expr(node, species_intervals)
    if node_type == AST_LOGICAL_NOT:
        return NotExpr(_ast_to_expr(node.getChild(0), species_intervals))
    if node_type == AST_CONSTANT_TRUE:
        return TRUE
    if node_type == AST_CONSTANT_FALSE:
        return FALSE
    if node_type == AST_NAME:
        return _level_expr(node.getName(), "==", 1, species_intervals)
    if node_type in _AST_COMPARISON_OPS:
        if node.getNumChildren() != 2:
            raise NotImplementedError(
                "SBML-Qual comparison node without exactly 2 children"
            )
        name_level = _name_and_level(node.getChild(0), node.getChild(1))
        if name_level is None:
            raise NotImplementedError(
                "SBML-Qual comparisons must be between a qualitative species and "
                "an integer level"
            )
        species, level = name_level
        return _level_expr(
            species, _AST_COMPARISON_OPS[node_type], level, species_intervals
        )
    raise NotImplementedError(f"Unsupported SBML-Qual MathML node type: {node_type}")


def _transition_expr(
    transition: Transition,
    species_intervals: dict[str, dict[int, _Interval] | None],
) -> BoolExpr:
    default_level: int = transition.getDefaultTerm().getResultLevel()
    fts = transition.getListOfFunctionTerms()
    terms_1: list[ASTNode] = []
    terms_0: list[ASTNode] = []
    for i in range(fts.size()):
        ft = fts.get(i)
        (terms_1 if ft.getResultLevel() == 1 else terms_0).append(ft.getMath())

    if len(terms_1) > 0 and len(terms_0) == 0:
        if default_level != 0:
            raise NotImplementedError(
                f"Transition '{transition.getId()}' has resultLevel-1 function "
                f"terms but a non-zero default level ({default_level})"
            )
        exprs = [_ast_to_expr(m, species_intervals) for m in terms_1]
        return exprs[0] if len(exprs) == 1 else OrExpr(exprs)
    if len(terms_0) > 0 and len(terms_1) == 0:
        if default_level != 1:
            raise NotImplementedError(
                f"Transition '{transition.getId()}' has resultLevel-0 function "
                f"terms but a non-one default level ({default_level})"
            )
        exprs = [_ast_to_expr(m, species_intervals) for m in terms_0]
        return NotExpr(exprs[0] if len(exprs) == 1 else OrExpr(exprs))
    if len(terms_1) == 0 and len(terms_0) == 0:
        return TRUE if default_level == 1 else FALSE
    raise NotImplementedError(
        f"Transition '{transition.getId()}' mixes resultLevel-0 and resultLevel-1 "
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
        for level, (lo, lo_open, hi, hi_open) in sorted(intervals.items())
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
    # Display
    # ==========================================================================
    def __str__(self: RegulatoryNetwork) -> str:
        return "\n".join(f"{n} <- {rule}" for n, rule in self.__rules.items())

    def __repr__(self: RegulatoryNetwork) -> str:
        return (
            f"{type(self).__name__}("
            + ", ".join(f"Rule({n};{repr(rule)}" for n, rule in self.__rules.items())
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
            for line in file.readlines():
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
        if not os.path.isfile(sbml):
            raise FileNotFoundError(f"Regulatory network SBML file not found: {sbml}")
        # ~ Read SBML-Qual file --------------------------------------------------
        sbmld: SBMLDocument = SBMLReader().readSBML(sbml)  # type: ignore
        sbmlm: Model = sbmld.getModel()  # type: ignore
        qual: QualModelPlugin | None = (
            sbmlm.getPlugin("qual") if sbmlm is not None else None
        )
        if qual is None:
            raise ValueError(f"Not an SBML-Qual (regulatory network) file: {sbml}")
        # ~ Species STATE intervals (thresholded species) / booleans (no notes) --
        species_intervals: dict[str, dict[int, _Interval] | None] = {}
        for i in range(qual.getNumQualitativeSpecies()):
            sp: QualitativeSpecies = qual.getQualitativeSpecies(i)
            species_intervals[sp.getId()] = (
                _parse_state_intervals(sp.getNotesString()) if sp.isSetNotes() else None
            )
        # ~ Initialize output ------------------------------------------------------
        rn = RegulatoryNetwork()
        # ~ Translate each transition into a Boolean rule -------------------------
        for i in range(qual.getNumTransitions()):
            transition: Transition = qual.getTransition(i)
            outputs = transition.getListOfOutputs()
            if outputs.size() != 1:
                raise NotImplementedError(
                    f"Transition '{transition.getId()}' does not have exactly "
                    "one output"
                )
            node: str = outputs.get(0).getQualitativeSpecies()
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


# ==============================================================================
# Main
# ==============================================================================
if __name__ == "__main__":
    ...