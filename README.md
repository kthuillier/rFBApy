# Small Python Package for Regulatory Flux Balance Analysis Simulation

Python 3 module `rFBApy` for rFBA simulations based on
> Covert, M. W., Schilling, C. H., and Palsson, B. (2001). Regulation of gene expression in flux balance models of metabolism. Journal of Theoretical Biology, 213(1):73–88.

## Install

Python >= 3.10.

Dependencies (pulled by pip): `python-libsbml`, `gurobipy`, `swiglpk`, `pandas`, `lark`, `networkx`, `clingo`, `merrinasp` (from GitHub), `pulp<4`.

LP solver: GLPK by default. Gurobi works too (`--lpsolver gurobi`) but needs a license.

### From the local repository

Install via pip:

```sh
pip install -e .
```

### From the remote GitHub repository

Install via pip:

```sh
pip install git+https://github.com/kthuillier/rFBApy
```

## Usage Examples

Toy model in `example/`:

```sh
bash example/simulate.sh   # -> example/output.csv
bash example/fixpoint.sh   # -> example/fixpoints.csv
```

### Python API Calls

```python
from rFBApy import simulate_rfba

df = simulate_rfba(
    "example/data/metabolic_network.sbml",
    "Growth",
    bnet="example/data/regulatory_network.bnet",
    concentrations={"Carbon1": 10, "Carbon2": 10, "Oxygen": 100},
    state={"RPcl": 1},
    bounds={"Tc1": (0, 10.5), "Tc2": (0, 10.5), "To2": (0, 15)},
    biomass=0.01,
    tau=0.01,
    iter=300,
)
df.to_csv("output.csv", index=False)
```

Other arguments:

- `sbmlqual`: SBML-qual regulatory network, instead of `bnet`.
- `mutations`: knock-outs, e.g. `{"R7": 0}`.
- `settings`: other variables read by the rules, e.g. `{"pH": 7}`.
- `compressed`, `lpsolver`, `lpeps`, `pfba`: same as the CLI options below.

Networks can also be passed as objects:

```python
from rFBApy.metabolic_network import MetabolicNetwork
from rFBApy.regulatory_network import RegulatoryNetwork

mn = MetabolicNetwork.read_sbml("example/data/metabolic_network.sbml")
bn = RegulatoryNetwork.load_bnet("example/data/regulatory_network.bnet")
df = simulate_rfba(mn, "Growth", bnet=bn, concentrations={"Carbon1": 10})
```

### Terminal

```sh
rfbapy --sbml example/data/metabolic_network.sbml \
    --bnet example/data/regulatory_network.bnet --obj Growth \
    --experiment example/data/experiment.json --out output.csv
```

- `--sbml`, `--obj`, `--experiment`, `--out`: required.
- `--bnet` or `--sbmlqual`: regulatory network. Optional.
- `--lpsolver glpk|gurobi`: default `glpk`.
- `--lpeps`: fluxes below this are set to 0. Default `1e-8`.
- `--no-pfba`: plain FBA instead of parsimonious FBA.
- `--compressed`: keep only the first and last row of each stable state.

### Output

One row per time step:

- `Time`, `biomass`
- `<metabolite> (met)`: external concentrations
- `<reaction> (reaction)`: fluxes
- `<node> (state)`: regulatory state (0/1)

### Experiment file

```json
{
    "SimulationParameters": {"initBiomass": 0.01, "timeStep": 0.01, "nSteps": 300},
    "initConcentrations": {"Carbon1": 10, "Carbon2": 10, "Oxygen": 100},
    "initRegs": {"RPcl": 1},
    "initGenes": {},
    "settings": {},
    "constraints": {
        "ko": [],
        "bounds": {"Tc1": [0, 10.5], "To2": [0, 15]}
    }
}
```

- `initConcentrations`: external metabolites. Missing ones are 0.
- `initRegs`, `initGenes`: initial regulatory state.
- `settings`: other variables read by the rules.
- `ko`: genes or reactions forced to 0.
- `bounds`: flux bounds, override the SBML ones.

Full example: `example/data/experiment.json`.

### Regulatory network (.bnet)

One rule per line: `node, rule`.

```text
RPcl, [Carbon1 > 0]
Tc2, !RPcl
R5a, !RPO2
```

- `&`, `|`, `!`, parentheses.
- `[X op v]` compares a metabolite concentration, a flux or a setting to `v`. `op` is one of `>`, `>=`, `<`, `<=`, `==`, `!=`.
- A rule named after a reaction switches it on/off.
- Gene-reaction rules (GPR) come from the SBML.

### Fixpoints

`rfbapy-fixpoint` looks for rFBA stable states (fixpoints) for a medium, a strain and abiotic factors.
No simulation.
Same inputs as `rfbapy`.
Solved in ASP with [merrinasp](https://github.com/kthuillier/merrinasp).

```sh
rfbapy-fixpoint --sbml example/data/metabolic_network.sbml \
    --bnet example/data/regulatory_network.bnet --obj Growth \
    --experiment example/data/experiment.json -n 0 --verify --out fixpoints.csv
```

- `-n N`: number of fixpoints, `0` for all. Default `1`.
- `--verify`: check each fixpoint with rFBApy. Adds `verified`, `optimum`, `duration` columns.
- `--no-optimal`: accept any growing flux vector, not only the FBA optimum.
- `--fix-initial-state`: impose `initRegs`/`initGenes` on the fixpoint. Ignored otherwise.
- `--cache FILE`: reuse work across runs on the same metabolic network.
- `--out`: optional. One row per fixpoint, same columns as the simulation (no `Time`, no `biomass`).
- `rfbapy-fixpoint --help` for the rest.

Secreted metabolites have no defined concentration: `NaN` in the CSV.

Python:

```python
from rFBApy import find_fixpoint

df = find_fixpoint(
    "example/data/metabolic_network.sbml",
    "Growth",
    bnet="example/data/regulatory_network.bnet",
    concentrations={"Carbon1": 10, "Carbon2": 10, "Oxygen": 100},
    optimal=True,
    n=0,
    verify=True,
)
```

Returns `None` when there is no fixpoint.
