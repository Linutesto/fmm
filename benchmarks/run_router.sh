#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python run_router_benchmark.py "$@"
python plot_router_results.py
echo "Done. See results/router_summary.json + fig_router_*.png"
