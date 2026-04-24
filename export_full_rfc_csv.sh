#!/usr/bin/env bash
set -euo pipefail

# Export complete RFC dataframe from cache into CSV.
# - includes all runs (--all-runs)
# - supports both labeled and legacy large-group cache layouts
# - forwards any extra CLI args to export_rfc_dataframe.py
#
# Example:
#   ./export_full_rfc_csv.sh
#   ./export_full_rfc_csv.sh --windows 1 3 5 10 --biases 0 1 2 --groups-counts 1000

OUT_CSV="cache/cons_evaluations/exports/rfc_results_full.csv"

python3 -m evaluation_frameworks.consensus_evaluation.evaluation.evaluations.print.export_rfc_dataframe \
  --all-runs \
  --out-csv "${OUT_CSV}" \
  "$@"

echo "Exported: ${OUT_CSV}"
