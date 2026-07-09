# ==============================================================================
# Import
# ==============================================================================
from __future__ import annotations

import argparse
import json
import pandas as pd # type: ignore

# ~ Custom modules
from rFBApy.rfba import simulate_rfba, ACCURACY
from rFBApy.fba import LP_SOLVERS, DEFAULT_SOLVER


# ==============================================================================
# Parser
# ==============================================================================
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run rFBA simulation from experiment JSON"
    )

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
        required=True,
        help="Output CSV file",
    )

    # ------------------------------------------------------------------
    # Optional
    # ------------------------------------------------------------------
    parser.add_argument(
        "--bnet", "-bn",
        type=str,
        default=None,
        help="Path to Boolean regulatory network",
    )

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
        help=f"Epsilon for to-zero clipping of LP values (Default: {ACCURACY})",
    )

    return parser


def load_experiment(path: str) -> dict:
    with open(path, "r") as f:
        exp = json.load(f)

    sim_params = exp.get("SimulationParameters", {})
    constraints = exp.get("constraints", {})

    return {
        "biomass": sim_params.get("initBiomass", 1e-2),
        "tau": sim_params.get("timeStep", 1e-2),
        "iter": sim_params.get("nSteps", 100),
        "concentrations": exp.get("initConcentrations", {}),
        "state": exp.get("initRegs", {}),
        # optional if you later distinguish genes/regs
        "genes": exp.get("initGenes", {}),
        "mutations": {ko: 0 for ko in constraints.get("ko", [])},
        "bounds": {r: tuple(v) for r, v in constraints.get("bounds", {}).items()},
        "settings": exp.get("settings", {}),
    }


def parse_args():
    parser = build_parser()
    args = parser.parse_args()

    exp = load_experiment(args.experiment)

    return {
        "sbml": args.sbml,
        "bnet": args.bnet,
        "obj": args.obj,
        "concentrations": exp["concentrations"],
        "state": exp["state"],
        "bounds": exp["bounds"],
        "mutations": exp["mutations"],
        "settings": exp["settings"],
        "biomass": exp["biomass"],
        "tau": exp["tau"],
        "iter": exp["iter"],
        "compressed": args.compressed,
        "lpsolver": args.lpsolver,
        "lpeps": args.lpeps,
        "out": args.out,
    }


# ==============================================================================
# Main
# ==============================================================================
def main() -> None:
    args = parse_args()

    sim_df: pd.DataFrame = simulate_rfba(
        sbml=args["sbml"],
        bnet=args["bnet"],
        obj=args["obj"],
        concentrations=args["concentrations"],
        state=args["state"],
        bounds=args["bounds"],
        mutations=args["mutations"],
        settings=args["settings"],
        biomass=args["biomass"],
        tau=args["tau"],
        iter=args["iter"],
        compressed=args["compressed"],
        lpsolver=args["lpsolver"],
        lpeps=args["lpeps"],
    )

    sim_df.to_csv(
        args["out"],
        sep=",",
        index=False,
    )

if __name__ == "__main__":
    main()
