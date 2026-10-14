#!/usr/bin/env bash

SD=$(realpath "$(dirname "$0")")

rfbapy-fixpoint --verify --stats --sbml ${SD}/data/metabolic_network.sbml --bnet ${SD}/data/regulatory_network.bnet --obj Growth --experiment ${SD}/data/experiment.json --out ${SD}/fixpoints.csv