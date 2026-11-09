# ==============================================================================
# Import
# ==============================================================================
from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import sys
import time
from collections import Counter
from importlib.metadata import version

import clingo
import pandas as pd  # type: ignore

from rFBApy.fba import DEFAULT_SOLVER, LP_SOLVERS

# ~ Custom modules
from rFBApy.rfba import (
    ACCURACY,
    BIOMASS,
    COLUMN_DURATION,
    COLUMN_OPTIMUM,
    COLUMN_VERIFIED,
    FIXPOINT_ACCURACY,
    OPTIMALITY,
    TAG_METABOLITE,
    TAG_REACTION,
    TAU,
    find_fixpoint,
    load_networks,
    simulate_rfba,
)


# ==============================================================================
# Parsers
# ==============================================================================
def _add_inputs(
    parser: argparse.ArgumentParser, out_required: bool, out_help: str
) -> None:
    # ~ Arguments shared by rfbapy and rfbapy-fixpoint
    # ------------------------------------------------------------------
    # Required
    # ------------------------------------------------------------------
    parser.add_argument(
        "--sbml", "-s",
        type=str,
        required=True,
        help="Path to SBML metabolic network",
    )

    parser.add_argument(
        "--obj",
        type=str,
        required=True,
        help="Objective reaction",
    )

    parser.add_argument(
        "--experiment", "-e",
        type=str,
        required=True,
        help="Path to experiment JSON file",
    )

    parser.add_argument(
        "--out", "-ou",
        type=str,
        required=out_required,
        default=None,
        help=out_help,
    )

    # ------------------------------------------------------------------
    # Optional
    # ------------------------------------------------------------------
    parser.add_argument(
        "--bnet", "-bn",
        type=str,
        default=None,
        help="Path to Boolean regulatory network (.bnet). Not required; if "
             "both --bnet and --sbmlqual are given, --bnet takes precedence.",
    )

    parser.add_argument(
        "--sbmlqual", "-sq",
        type=str,
        default=None,
        help="Path to SBML-Qual regulatory network. Not required; ignored if "
             "--bnet is also given.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run rFBA simulation from experiment JSON"
    )
    _add_inputs(parser, out_required=True, out_help="Output CSV file")

    parser.add_argument(
        "--compressed",
        action="store_true",
        help="Store only transition states",
    )

    parser.add_argument(
        "--lpsolver",
        type=str,
        choices=LP_SOLVERS,
        default=DEFAULT_SOLVER,
        help=f"LP solver ({', '.join(LP_SOLVERS)})",
    )

    parser.add_argument(
        "--lpeps",
        type=float,
        default=ACCURACY,
        help="Values within this of 0 are set to 0, for the float solvers "
             f"(glpk, gurobi); rational is exact (Default: {ACCURACY})",
    )

    parser.add_argument(
        "--pfba",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use parsimonious FBA (minimize total flux at the optimal "
             "objective value) instead of plain FBA (Default: True)",
    )

    return parser


def _count(text: str) -> int:
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError("must be 0 (all fixpoints) or more")
    return value


def build_fixpoint_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rfbapy-fixpoint",
        description="Find fixpoints of a regulated metabolic network (rFBA without simulation)",
    )
    _add_inputs(
        parser,
        out_required=False,
        out_help="Output CSV file, one row per fixpoint (Default: none)",
    )

    parser.add_argument(
        "--lpsolver",
        type=str,
        choices=LP_SOLVERS,
        default=DEFAULT_SOLVER,
        help=f"LP solver ({', '.join(LP_SOLVERS)}), of merrinasp and of the "
             f"FBA (Default: {DEFAULT_SOLVER})",
    )

    parser.add_argument(
        "--lpeps",
        type=float,
        default=FIXPOINT_ACCURACY,
        help=f"Smallest flux counted as non-zero (Default: {FIXPOINT_ACCURACY})",
    )

    # ------------------------------------------------------------------
    # Fixpoints only
    # ------------------------------------------------------------------
    parser.add_argument(
        "-n", "--n",
        type=_count,
        default=1,
        help="Number of fixpoints to look for, 0 for all of them (Default: 1)",
    )

    parser.add_argument(
        "--optimal",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require the flux vector to reach the FBA optimum of its "
             "regulatory state (Default: True)",
    )

    parser.add_argument(
        "--opttol",
        type=float,
        default=OPTIMALITY,
        help=f"Relative tolerance on that optimum (Default: {OPTIMALITY})",
    )

    parser.add_argument(
        "--bisection",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Find the conflicts of the linear program by bisection rather "
             "than with merrinasp's own analysis (Default: True)",
    )

    parser.add_argument(
        "--fix-initial-state",
        action="store_true",
        help="Impose the initial state of the experiment (initRegs, initGenes) "
             "on the fixpoint",
    )

    parser.add_argument(
        "--verify",
        action="store_true",
        help="Check every fixpoint with rFBApy, without ASP",
    )

    parser.add_argument(
        "--cache",
        type=str,
        default=None,
        help="File of cuts, FBA optima and fixpoints shared with other runs on "
             "the same metabolic network: read, then appended to",
    )

    parser.add_argument(
        "--stats",
        action="store_true",
        help="Print the cost of the search: LP calls, FBA, cuts, reuse",
    )

    return parser


# ==============================================================================
# Experiment
# ==============================================================================
def load_experiment(path: str) -> dict:
    with open(path, "r") as f:
        exp = json.load(f)

    sim_params = exp.get("SimulationParameters", {})
    constraints = exp.get("constraints", {})

    return {
        "biomass": sim_params.get("initBiomass", BIOMASS),
        "tau": sim_params.get("timeStep", TAU),
        "iter": sim_params.get("nSteps", 100),
        "concentrations": exp.get("initConcentrations", {}),
        "state": exp.get("initRegs", {}) | exp.get("initGenes", {}),
        "mutations": {ko: 0 for ko in constraints.get("ko", [])},
        "bounds": {r: tuple(v) for r, v in constraints.get("bounds", {}).items()},
        "settings": exp.get("settings", {}),
    }


# ==============================================================================
# Fixpoint Output
# ==============================================================================
def _answers(
    df: pd.DataFrame, obj: str, problems: list[list[str]], verify: bool
) -> None:
    # ~ One block per fixpoint, as clingo prints its answers
    for index in range(len(df)):
        row = df.iloc[index]
        secreted = [
            c[: -len(TAG_METABOLITE) - 1]
            for c in df.columns
            if c.endswith(" " + TAG_METABOLITE) and isinstance(row[c], float) and math.isnan(row[c])
        ]
        print(f"Answer: {index + 1}")
        print(f"Growth: {obj}={float(row[f'{obj} {TAG_REACTION}']):.6g}")
        print("Secreted: " + (" ".join(secreted) if secreted else "none"))
        if verify:
            print(
                "Check: " + ("OK" if bool(row[COLUMN_VERIFIED]) else "FAILED")
                + f" | rFBA optimum {float(row[COLUMN_OPTIMUM]):.6g}"
                + f" | holds {float(row[COLUMN_DURATION]):g}"
            )
            for problem in problems[index] if index < len(problems) else []:
                print(f"  {problem}")


def _printed(text: str) -> tuple[list[str], list[list[str]]]:
    # ~ Warnings, and failed conditions per fixpoint, from what find_fixpoint printed
    warnings: list[str] = []
    problems: list[list[str]] = []
    for line in text.splitlines():
        if line.startswith("Warning:"):
            warnings.append(line)
        elif line.startswith("Fixpoint check with rFBApy") and "nothing to check" not in line:
            problems.append([])
        elif line.startswith("  - ") and problems:
            problems[-1].append(line[2:])
    return warnings, problems


def _statistics(
    args: argparse.Namespace, count: int, stats: Counter[str], total: float, reading: float,
    cpu: float,
) -> None:
    # ~ The closing block of clingo
    def line(key: str, value: str) -> None:
        print(f"{key:<13}: {value}")

    line("Models", f"{count}" + ("" if args.n == 0 or count < args.n else "+"))
    if args.optimal:
        line("Cuts", f"{stats['cuts added']}" + (
            f" (+{stats['cuts imported']} from the cache)" if stats["cuts imported"] else ""
        ))
    line("Calls", f"{stats['asp solves']}")
    line("Time", f"{total:.3f}s (Reading: {reading:.2f}s Solving: {total - reading:.2f}s)")
    line("CPU Time", f"{cpu:.3f}s")
    if args.stats:
        print()
        line("Choices", f"{stats['choices']}")
        line("Conflicts", f"{stats['conflicts']}")
        line("LP conflicts", f"{stats['lp conflicts']} (analysed by "
                             + ("bisection" if args.bisection else "merrinasp") + ")")
        line("LP calls", f"{stats['lp calls']} ({stats['lp seconds']:.2f}s)")
        line("FBA", f"{stats['fba solved']} solved, {stats['fba reused']} reused")
        reused = []
        if stats["fixpoints reused"]:
            reused.append("fixpoint of the instance without mutation")
        if stats["no growth without regulation"]:
            reused.append("no growth even with every reaction available")
        if reused:
            line("Shortcut", ", ".join(reused))
        line("LP solver", args.lpsolver)


# ==============================================================================
# Main
# ==============================================================================
def main(argv: list[str] | None = None) -> None:
    # ~ rfbapy
    args = build_parser().parse_args(argv)
    experiment = load_experiment(args.experiment)

    sim_df: pd.DataFrame = simulate_rfba(
        sbml=args.sbml,
        bnet=args.bnet,
        sbmlqual=args.sbmlqual,
        obj=args.obj,
        concentrations=experiment["concentrations"],
        state=experiment["state"],
        bounds=experiment["bounds"],
        mutations=experiment["mutations"],
        settings=experiment["settings"],
        biomass=experiment["biomass"],
        tau=experiment["tau"],
        iter=experiment["iter"],
        compressed=args.compressed,
        lpsolver=args.lpsolver,
        lpeps=args.lpeps,
        pfba=args.pfba,
    )

    sim_df.to_csv(
        args.out,
        sep=",",
        index=False,
    )


def main_fixpoint(argv: list[str] | None = None) -> int:
    # ~ rfbapy-fixpoint: 0 whether fixpoints are found or not, 1 on an error
    args = build_fixpoint_parser().parse_args(argv)
    # ~ Imported here: only rfbapy-fixpoint needs merrinasp
    from rFBApy.fixpoint import solver
    from rFBApy.fixpoint.cache import Cache

    start, cpu = time.perf_counter(), time.process_time()
    print(
        f"rfbapy-fixpoint (rFBApy {version('rFBApy')}, clingo {clingo.__version__}, "
        f"merrinasp {version('merrinasp')})"
    )
    regulation = args.bnet if args.bnet is not None else args.sbmlqual
    print("Reading from " + " ".join(p for p in (args.sbml, regulation, args.experiment) if p))
    try:
        experiment = load_experiment(args.experiment)
        mn, bn = load_networks(args.sbml, args.bnet, args.sbmlqual)
        cache = Cache(args.cache)
    except Exception as error:  # noqa: BLE001
        print(f"*** ERROR: (rfbapy-fixpoint): {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    reading = time.perf_counter() - start

    print("Solving...", flush=True)
    solver.BISECTION = args.bisection
    printed = io.StringIO()
    try:
        with contextlib.redirect_stdout(printed):
            df = find_fixpoint(
                mn, args.obj, bnet=bn,
                concentrations=experiment["concentrations"],
                state=experiment["state"] if args.fix_initial_state else {},
                bounds=experiment["bounds"], mutations=experiment["mutations"],
                settings=experiment["settings"], lpsolver=args.lpsolver, lpeps=args.lpeps,
                verify=args.verify, optimal=args.optimal,
                opttol=args.opttol, n=args.n, cache=cache,
                biomass=experiment["biomass"], tau=experiment["tau"],
            )
    except Exception as error:  # noqa: BLE001
        sys.stderr.write(printed.getvalue())
        print(f"*** ERROR: (rfbapy-fixpoint): {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    warnings, problems = _printed(printed.getvalue())
    for warning in warnings:
        print(warning, file=sys.stderr)

    count = 0 if df is None else len(df)
    if df is not None:
        _answers(df, args.obj, problems, args.verify)
    print("SATISFIABLE" if count else "UNSATISFIABLE")
    if args.out is not None:
        if df is not None:
            df.to_csv(args.out, sep=",", index=False)
        else:
            print(f"rfbapy-fixpoint: no fixpoint, {args.out} not written", file=sys.stderr)
    print()
    _statistics(
        args, count, cache.stats, time.perf_counter() - start, reading, time.process_time() - cpu
    )
    return 0


if __name__ == "__main__":
    main()
