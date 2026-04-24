#!/usr/bin/env bash
set -euo pipefail

# remote
REMOTE_USER="cross-bit"
REMOTE_HOST="groups-eval"
REMOTE_BASE="/home/cross-bit/analysis"

# local (spouštíš z rootu local analysis)
LOCAL_BASE="$(pwd)"

echo "[sync] local:  $LOCAL_BASE"
echo "[sync] remote: ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_BASE}"

# 1) pouze PKL z cache/cons_evaluations (zachová strukturu)
rsync -avz --prune-empty-dirs \
  --include='*/' \
  --include='*.pkl' \
  --exclude='*' \
  "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_BASE}/cache/cons_evaluations/" \
  "${LOCAL_BASE}/cache/cons_evaluations/"

# 2) table script
rsync -avz \
  "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_BASE}/evaluation_frameworks/consensus_evaluation/evaluation/evaluations/print/table_rfc_large_group_size_comparisions.py" \
  "${LOCAL_BASE}/evaluation_frameworks/consensus_evaluation/evaluation/evaluations/print/"

# 3) všechny large eval skripty
rsync -avz \
  "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_BASE}/evaluation_frameworks/consensus_evaluation/evaluation/evaluations/larger_group_evaluations/" \
  "${LOCAL_BASE}/evaluation_frameworks/consensus_evaluation/evaluation/evaluations/larger_group_evaluations/"

echo "[sync] done"