#!/usr/bin/env bash

SD=$(realpath "$(dirname "$0")")

rfbapy --sbml ${SD}/data/metabolic_network.sbml --bnet ${SD}/data/regulatory_network.bnet --obj Growth --experiment ${SD}/data/experiment.json --out ${SD}/output.csv