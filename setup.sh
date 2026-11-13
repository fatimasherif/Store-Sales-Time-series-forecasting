#!/usr/bin/env bash
# One-time project setup: venv, packages, Jupyter kernel, folders.
# Run from the project root:  bash setup.sh
set -euo pipefail
cd "$(dirname "$0")"

# 1. Virtual environment with Python 3.12
if [ ! -d .venv ]; then
    python3.12 -m venv .venv
fi
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# 2. Register the venv as a Jupyter kernel (pick it in VS Code)
python -m ipykernel install --user --name store-sales --display-name "Python 3.12 (store-sales)"

# 3. Folder structure (.gitkeep so empty folders exist in git)
for d in data/raw data/processed/backtest_preds notebooks src reports/figures submissions; do
    mkdir -p "$d"
done
touch data/raw/.gitkeep data/processed/.gitkeep reports/figures/.gitkeep submissions/.gitkeep

python --version
echo "Setup done. Next: download the data (see README, 'How to reproduce')."
