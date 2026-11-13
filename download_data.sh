#!/usr/bin/env bash
# Download the competition files into data/raw and make them read-only.
# Before running:
#   1. Accept the competition rules on the Kaggle website (Join Competition).
#   2. Authenticate the Kaggle CLI, either:
#        export KAGGLE_API_TOKEN=<token from kaggle.com/settings/api>
#      or save the token in ~/.kaggle/access_token (chmod 600).
# Run from the project root:  bash download_data.sh
set -euo pipefail
cd "$(dirname "$0")"
source .venv/bin/activate

COMP=store-sales-time-series-forecasting
kaggle competitions download "$COMP" -p data/raw --unzip

chmod a-w data/raw/*.csv
ls -lh data/raw
