# ==============================================================================
# Imports
# ==============================================================================
from __future__ import annotations

import operator
from abc import ABC, abstractmethod
from typing import Any, Callable

from lark import Lark, Transformer

_OPS: dict[str, Callable[[float, float], bool]] = {
    ">=": operator.ge,
    ">": operator.gt,
    "<=": operator.le,
    "<": operator.lt,
    "==": operator.eq,
    "!=": operator.ne,
}

_OPS_NEG: dict[str, str] = {
    ">=": "<",
    ">": "<=",
    "<=": ">",
    "<": ">=",
    "==": "!=",
    "!=": "==",
}


# ==============================================================================
# AST for Boolean Rules with Thresholds
# ==============================================================================
# ~ Abstract class: BoolExpr
class BoolExpr(ABC):
    __slots__ = ("_vars", "_threshold", "_signed_vars")

    @abstractmethod
    def compile(self: BoolExpr) -> Callable[[dict[str, bool | int | float]], bool]: ...

    @abstractmethod
    def simplify(self: BoolExpr) -> BoolExpr: ...

    def to_dnf(self: BoolExpr) -> OrExpr:
        return to_dnf(self)

    def to_cnf(self: BoolExpr) -> AndExpr:
        return to_cnf(self)
    
    def is_constant(self: BoolExpr) -> bool:
        return isinstance(self.simplify(), (TrueExpr, FalseExpr))

    @abstractmethod
    def __call__(self: BoolExpr, context: dict[str, bool | int | float]) -> bool: ...

    @abstractmethod
    def __str__(self: BoolExpr) -> str: ...

    @abstractmethod
    def __repr__(self: BoolExpr) -> str: ...

    @abstractmethod
    def __eq__(self: BoolExpr, other: Any) -> bool: ...

    @abstractmethod
    def __hash__(self: BoolExpr) -> int: ...

    @property
    @abstractmethod
    def variables(self: BoolExpr) -> frozenset[str]: ...

    @property
    @abstractmethod
    def signed_variables(self: BoolExpr) -> frozenset[tuple[str, int]]: ...

    @property
    @abstractmethod
    def thresholds(self: BoolExpr) -> frozenset[tuple[str, float]]: ...


# ~ Boolean nodes
class AndExpr(BoolExpr):
    __slots__ = ("items", "_hash")

    def __init__(self, items: list[BoolExpr]):
        self.items = items
        self._vars = None
        self._signed_vars = None
        self._threshold = None
        self._hash = None

    def compile(self: AndExpr) -> Callable[[dict[str, bool | int | float]], bool]:
        subfns = [x.compile() for x in self.items]
        if len(subfns) == 2:
            f0, f1 = subfns
            return lambda state: f0(state) and f1(state)
        return lambda state: all(f(state) for f in subfns)

    def simplify(self: AndExpr) -> BoolExpr:
        eval_items = []
        seen: set[BoolExpr] = set()
        for item in self.items:
            eval_item = item.simplify()
            if eval_item is FALSE:
                return FALSE
            if eval_item is TRUE:
                continue
            if eval_item in seen:
                continue
            seen.add(eval_item)
            eval_items.append(eval_item)
        if len(eval_items) == 0:
            return TRUE
        if len(eval_items) == 1:
            return eval_items[0]
        return AndExpr(eval_items)

    def __call__(self: AndExpr, context: dict[str, bool | int | float]) -> bool:
        return all(x(context) for x in self.items)

    def __iter__(self: AndExpr):
        yield from self.items

    def __str__(self: AndExpr) -> str:
        return "(" + " & ".join(str(x) for x in self.items) + ")"

    def __repr__(self: AndExpr) -> str:
        return f"{type(self).__name__}({', '.join(repr(x) for x in self.items)})"

    def __eq__(self: AndExpr, other: Any):
        if not isinstance(other, AndExpr) or len(self.items) != len(other.items):
            return False
        return all(sx == ox for sx, ox in zip(self.items, other.items))

    def __hash__(self):
        if self._hash is None:
            self._hash = hash(("AndExpr", frozenset(self.items)))
        return self._hash

    @property
    def variables(self: AndExpr) -> frozenset[str]:
        if self._vars is None:
            self._vars = frozenset().union(*(x.variables for x in self.items))
        return self._vars

    @property
    def signed_variables(self: AndExpr) -> frozenset[tuple[str, int]]:
        if self._signed_vars is None:
            acc: dict[str, int] = {}
            for s in self.items:
                for var, sign in s.signed_variables:
                    if var not in acc:
                        acc[var] = sign
                    elif acc[var] != sign:
                        acc[var] = 0
            self._signed_vars = frozenset(acc.items())
        return self._signed_vars

    @property
    def thresholds(self: AndExpr) -> frozenset[tuple[str, float]]:
        if self._threshold is None:
            self._threshold = frozenset().union(*(x.thresholds for x in self.items))
        return self._threshold


# ~ Boolean nodes
class OrExpr(BoolExpr):
    __slots__ = ("items", "_hash")

    def __init__(self: OrExpr, items: list[BoolExpr]):
        self.items = items
        self._vars = None
        self._signed_vars = None
        self._threshold = None

    def compile(self: OrExpr) -> Callable[[dict[str, bool | int | float]], bool]:
        subfns = [x.compile() for x in self.items]
        if len(subfns) == 2:
            f0, f1 = subfns
            return lambda state: f0(state) or f1(state)
        return lambda state: any(f(state) for f in subfns)

    def simplify(self: OrExpr) -> BoolExpr:
        eval_items = []
        seen: set[BoolExpr] = set()
        for item in self.items:
            eval_item = item.simplify()
            if eval_item is TRUE:
                return TRUE
            if eval_item is FALSE:
                continue
            if eval_item in seen:
                continue
            seen.add(eval_item)
            eval_items.append(eval_item)
        if len(eval_items) == 0:
            return FALSE
        if len(eval_items) == 1:
            return eval_items[0]
        return OrExpr(eval_items)

    def __call__(self: OrExpr, context: dict[str, bool | int | float]) -> bool:
        return any(x(context) for x in self.items)

    def __iter__(self: OrExpr):
        yield from self.items

    def __str__(self: OrExpr) -> str:
        return "(" + " | ".join(str(x) for x in self.items) + ")"

    def __repr__(self: OrExpr) -> str:
        return f"{type(self).__name__}({', '.join(repr(x) for x in self.items)})"

    def __eq__(self: OrExpr, other: Any):
        if not isinstance(other, OrExpr) or len(self.items) != len(other.items):
            return False
        return all(sx == ox for sx, ox in zip(self.items, other.items))

    def __hash__(self):
        if self._hash is None:
            self._hash = hash(("OrExpr", frozenset(self.items)))
        return self._hash

    @property
    def variables(self: OrExpr) -> frozenset[str]:
        if self._vars is None:
            self._vars = frozenset().union(*(x.variables for x in self.items))
        return self._vars

    @property
    def signed_variables(self: OrExpr) -> frozenset[tuple[str, int]]:
        if self._signed_vars is None:
            acc: dict[str, int] = {}
            for s in self.items:
                for var, sign in s.signed_variables:
                    if var not in acc:
                        acc[var] = sign
                    elif acc[var] != sign:
                        acc[var] = 0
            self._signed_vars = frozenset(acc.items())
        return self._signed_vars

    @property
    def thresholds(self: OrExpr) -> frozenset[tuple[str, float]]:
        if self._threshold is None:
            self._threshold = frozenset().union(*(x.thresholds for x in self.items))
        return self._threshold


# ~ Boolean nodes
class NotExpr(BoolExpr):
    __slots__ = ("item",)

    def __init__(self: NotExpr, item: BoolExpr):
        self.item: BoolExpr = item
        self._vars = None
        self._signed_vars = None
        self._threshold = None

    def compile(self: NotExpr) -> Callable[[dict[str, bool | int | float]], bool]:
        subfn = self.item.compile()
        return lambda state: not subfn(state)

    def simplify(self: NotExpr) -> BoolExpr:
        eval_expr = self.item.simplify()
        if eval_expr is TRUE:
            return FALSE
        if eval_expr is FALSE:
            return TRUE
        if isinstance(eval_expr, ThresholdExpr):
            return ThresholdExpr(
                eval_expr.variable,
                _OPS_NEG[eval_expr.op],
                eval_expr.value,
            )
        return NotExpr(eval_expr)

    def __call__(self: NotExpr, context: dict[str, bool | int | float]) -> bool:
        return not self.item(context)

    def __str__(self: NotExpr) -> str:
        return "!" + str(self.item)

    def __repr__(self: NotExpr) -> str:
        return f"{type(self).__name__}({repr(self.item)})"

    def __eq__(self: NotExpr, other: Any):
        if not isinstance(other, NotExpr):
            return False
        return self.item == other.item

    def __hash__(self):
        return hash(("NotExpr", self.item))

    @property
    def variables(self) -> frozenset[str]:
        return self.item.variables

    @property
    def signed_variables(self: NotExpr) -> frozenset[tuple[str, int]]:
        if self._signed_vars is None:
            self._signed_vars = frozenset(
                [(x, -s if s != 0 else 0) for (x, s) in self.item.signed_variables]
            )
        return self._signed_vars

    @property
    def thresholds(self) -> frozenset[tuple[str, float]]:
        return self.item.thresholds


class VarExpr(BoolExpr):
    __slots__ = ("name",)

    def __init__(self: VarExpr, name: str):
        self.name: str = name
        self._vars = None
        self._signed_vars = None
        self._threshold = None

    def compile(self: VarExpr) -> Callable[[dict[str, bool | int | float]], bool]:
        var = self.name
        return lambda state: bool(state[var])

    def simplify(self: VarExpr) -> VarExpr:
        return VarExpr(self.name)

    def __call__(self: VarExpr, context: dict[str, bool | int | float]) -> bool:
        return bool(context[self.name])

    def __str__(self: VarExpr) -> str:
        return self.name

    def __repr__(self: VarExpr) -> str:
        return f"{type(self).__name__}({str(self)})"

    def __eq__(self: VarExpr, other: Any):
        if not isinstance(other, VarExpr):
            return False
        return self.name == other.name

    def __hash__(self):
        return hash(("VarExpr", self.name))

    @property
    def variables(self) -> frozenset[str]:
        if self._vars is None:
            self._vars = frozenset([self.name])
        return self._vars

    @property
    def signed_variables(self: VarExpr) -> frozenset[tuple[str, int]]:
        if self._signed_vars is None:
            self._signed_vars = frozenset([(self.name, 1)])
        return self._signed_vars

    @property
    def thresholds(self) -> frozenset[tuple[str, float]]:
        if self._threshold is None:
            self._threshold = frozenset()
        return self._threshold


class ThresholdExpr(BoolExpr):
    __slots__ = ("variable", "op", "value")

    def __init__(self: ThresholdExpr, variable: VarExpr, op: str, value: float):
        self.variable: VarExpr = variable
        self.op: str = op
        self.value: float = value
        self._vars = None
        self._signed_vars = None
        self._threshold = None

    def compile(self: ThresholdExpr) -> Callable[[dict[str, bool | int | float]], bool]:
        var = self.variable.name
        opfn = _OPS[self.op]
        val = self.value
        return lambda state: opfn(state[var], val)

    def simplify(self: ThresholdExpr) -> ThresholdExpr:
        return ThresholdExpr(
            self.variable.simplify(),
            self.op,
            self.value,
        )

    def __call__(self: ThresholdExpr, context: dict[str, bool | int | float]) -> bool:
        return _OPS[self.op](context[self.variable.name], self.value)

    def __str__(self: ThresholdExpr) -> str:
        return f"[{str(self.variable)} {self.op} {str(self.value)}]"

    def __repr__(self: ThresholdExpr) -> str:
        return f"{type(self).__name__}({str(self)})"

    def __eq__(self: ThresholdExpr, other: Any):
        if not isinstance(other, ThresholdExpr):
            return False
        return (
            self.variable == other.variable
            and self.op == other.op
            and self.value == other.value
        )

    def __hash__(self):
        return hash(("ThresholdExpr", self.variable, self.op, self.value))

    @property
    def variables(self: ThresholdExpr) -> frozenset[str]:
        return self.variable.variables

    @property
    def signed_variables(self: ThresholdExpr) -> frozenset[tuple[str, int]]:
        if self._signed_vars is None:
            s = 0
            if self.op == "<=":
                s = -1
            if self.op == ">=":
                s = 1
            self._signed_vars = frozenset([(self.variable.name, s)])
        return self._signed_vars

    @property
    def thresholds(self: ThresholdExpr) -> frozenset[tuple[str, float]]:
        if self._threshold is None:
            self._threshold = frozenset([(self.variable.name, self.value)])
        return self._threshold


class TrueExpr(BoolExpr):
    __slots__ = ()

    def __init__(self: TrueExpr):
        self._vars = frozenset()
        self._signed_vars = frozenset()
        self._threshold = frozenset()

    def compile(self: TrueExpr) -> Callable[[dict[str, bool | int | float]], bool]:
        return lambda _: True

    def simplify(self: TrueExpr) -> BoolExpr:
        return self

    def __call__(self: TrueExpr, context: dict[str, bool | int | float]) -> bool:
        return True

    def __str__(self: TrueExpr) -> str:
        return "1"

    def __repr__(self: TrueExpr) -> str:
        return f"{type(self).__name__}"

    def __eq__(self: TrueExpr, other: Any):
        return isinstance(other, TrueExpr)

    def __hash__(self):
        return hash("TrueExpr")

    @property
    def variables(self: TrueExpr) -> frozenset[str]:
        return self._vars

    @property
    def signed_variables(self: TrueExpr) -> frozenset[tuple[str, int]]:
        return self._signed_vars

    @property
    def thresholds(self: TrueExpr) -> frozenset[tuple[str, float]]:
        return self._threshold


class FalseExpr(BoolExpr):
    __slots__ = ()

    def __init__(self: FalseExpr):
        self._vars = frozenset()
        self._signed_vars = frozenset()
        self._threshold = frozenset()

    def compile(self: FalseExpr) -> Callable[[dict[str, bool | int | float]], bool]:
        return lambda _: False

    def simplify(self: FalseExpr) -> BoolExpr:
        return self

    def __call__(self: FalseExpr, context: dict[str, bool | int | float]) -> bool:
        return False

    def __str__(self: FalseExpr) -> str:
        return "0"

    def __repr__(self: FalseExpr) -> str:
        return f"{type(self).__name__}"

    def __eq__(self: FalseExpr, other: Any):
        return isinstance(other, FalseExpr)

    def __hash__(self):
        return hash("FalseExpr")

    @property
    def variables(self: FalseExpr) -> frozenset[str]:
        return self._vars

    @property
    def signed_variables(self: BoolExpr) -> frozenset[tuple[str, int]]:
        return self._signed_vars

    @property
    def thresholds(self: FalseExpr) -> frozenset[tuple[str, float]]:
        return self._threshold


TRUE = TrueExpr()
FALSE = FalseExpr()


# ==============================================================================
# Helper function for DNF and CNF forms
# ==============================================================================
# ------------------------------------------------------------------------------
# NNF -- (De Morgan)
# ------------------------------------------------------------------------------
def to_nnf(node: BoolExpr) -> BoolExpr:
    if isinstance(node, NotExpr):
        inner = node.item
        if isinstance(inner, NotExpr):
            return to_nnf(inner.item)
        if isinstance(inner, AndExpr):
            return OrExpr([to_nnf(NotExpr(x)) for x in inner.items])
        if isinstance(inner, OrExpr):
            return AndExpr([to_nnf(NotExpr(x)) for x in inner.items])
        if inner is TRUE:
            return FALSE
        if inner is FALSE:
            return TRUE
        return NotExpr(inner)
    if isinstance(node, AndExpr):
        return AndExpr([to_nnf(x) for x in node.items])
    if isinstance(node, OrExpr):
        return OrExpr([to_nnf(x) for x in node.items])
    return node


# ------------------------------------------------------------------------------
# List of clauses
# ------------------------------------------------------------------------------
def _complement(lit: BoolExpr) -> BoolExpr:
    return lit.item if isinstance(lit, NotExpr) else NotExpr(lit)


def _dnf_clauses(node: BoolExpr) -> list[frozenset[BoolExpr]]:
    if node is TRUE:
        return [frozenset()]
    if node is FALSE:
        return []
    if isinstance(node, OrExpr):
        result = []
        for x in node.items:
            result.extend(_dnf_clauses(x))
        return result
    if isinstance(node, AndExpr):
        acc = [frozenset()]
        for x in node.items:
            sub = _dnf_clauses(x)
            acc = [a | b for a in acc for b in sub]
        return acc
    return [frozenset([node])]


def _cnf_clauses(node: BoolExpr) -> list[frozenset[BoolExpr]]:
    if node is TRUE:
        return []
    if node is FALSE:
        return [frozenset()]
    if isinstance(node, AndExpr):
        result = []
        for x in node.items:
            result.extend(_cnf_clauses(x))
        return result
    if isinstance(node, OrExpr):
        acc = [frozenset()]
        for x in node.items:
            sub = _cnf_clauses(x)
            acc = [a | b for a in acc for b in sub]
        return acc
    return [frozenset([node])]


# ------------------------------------------------------------------------------
# Simplifying
# ------------------------------------------------------------------------------
def _remove_trivial_clauses(clauses: list[frozenset[BoolExpr]]):
    return [c for c in clauses if not any(_complement(k) in c for k in c)]


def _remove_subsumed(clauses: list[frozenset[BoolExpr]]):
    uniq = sorted(set(clauses), key=len)
    result = []
    for i, c in enumerate(uniq):
        if not any(other <= c and other != c for other in uniq[:i]):
            result.append(c)
    return result


# ------------------------------------------------------------------------------
# Tree-based representations
# ------------------------------------------------------------------------------
def _clauses_to_dnf_tree(clauses: list[frozenset[BoolExpr]]) -> OrExpr:
    if not clauses:
        return OrExpr([FALSE])
    terms = []
    for clause in clauses:
        if not clause:
            return OrExpr([TRUE])
        lits = list(clause)
        terms.append(lits[0] if len(lits) == 1 else AndExpr(lits))
    return OrExpr(terms)


def _clauses_to_cnf_tree(clauses: list[frozenset[BoolExpr]]) -> AndExpr:
    if not clauses:
        return AndExpr([TRUE])
    terms = []
    for clause in clauses:
        if not clause:
            return AndExpr([FALSE])
        lits = list(clause)
        terms.append(lits[0] if len(lits) == 1 else OrExpr(lits))
    return AndExpr(terms)


def to_dnf(node: BoolExpr) -> OrExpr:
    nnf = to_nnf(node.simplify())
    clauses = _remove_subsumed(_remove_trivial_clauses(_dnf_clauses(nnf)))
    return _clauses_to_dnf_tree(clauses)


def to_cnf(node: BoolExpr) -> AndExpr:
    nnf = to_nnf(node.simplify())
    clauses = _remove_subsumed(_remove_trivial_clauses(_cnf_clauses(nnf)))
    return _clauses_to_cnf_tree(clauses)


# ==============================================================================
# Parser
# ==============================================================================
# ~ BNET Grammar ---------------------------------------------------------------
bnet_grammar: str = r"""
    expr: or_expr
    or_expr: and_expr ("|" and_expr)*
    and_expr: not_expr ("&" not_expr)*
    not_expr: NOT* (threshold_expr | "(" expr ")")
    threshold_expr: var_expr | CONSTANT | "[" var_expr OP FLOAT "]"
    var_expr: VARID
    NOT: "!"
    VARID: /[a-zA-Z_][a-zA-Z0-9_]*/
    CONSTANT: "0" | "1"
    OP: ">=" | "<=" | "==" | "!=" | ">" | "<"
    FLOAT: /-?[0-9]+(\.[0-9]+)?(e[0-9]+)?/

    %import common.WS
    %ignore WS
"""


# ~ Lark AST Transformer -------------------------------------------------------
class BnetExprTransformer(Transformer):
    def expr(self, items):
        return items[0]

    def or_expr(self, items):
        if len(items) == 1:
            return items[0]
        return OrExpr(items)

    def and_expr(self, items):
        if len(items) == 1:
            return items[0]
        return AndExpr(items)

    def not_expr(self, items):
        return items[-1] if (len(items) - 1) % 2 == 0 else NotExpr(items[-1])

    def threshold_expr(self, items):
        if len(items) == 1:
            if isinstance(items[0], str) and items[0] in ("0", "1"):
                return TRUE if items[0] == "1" else FALSE
            return items[0]

        var = items[0]
        op = items[1]
        value = items[2]
        return ThresholdExpr(var, op, value)

    def var_expr(self, items):
        return VarExpr(items[0])

    def VARID(self, token):
        return str(token)

    def OP(self, token):
        return str(token)

    def FLOAT(self, token):
        return float(token)


# ~ Initialize lark parser -----------------------------------------------------
parser: Lark = Lark(
    bnet_grammar,
    start="expr",
    parser="lalr",
    transformer=BnetExprTransformer(),
)


# ~ Initialize lark parser -----------------------------------------------------
def parse(rule: str) -> BoolExpr:
    return parser.parse(rule)  # type: ignore


# ==============================================================================
# Main
# ==============================================================================
if __name__ == "__main__":
    ...
