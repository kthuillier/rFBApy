#!/usr/bin/env bash
set -euo pipefail

SD=$(realpath "$(dirname "$0")")
VENV=${SD}/../.venv

rm -rf "${VENV}"
python3 -m venv "${VENV}"
source "${VENV}/bin/activate"

pip install --upgrade pip
pip install -e "${SD}/..[dev]"
