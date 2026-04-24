#!/usr/bin/env bash
# Single entry point for consensus evaluation orchestration (bash).
# Usage: ./run_consensus.sh <command> [args...]
# Environment variables match the former per-script defaults; see ./run_consensus.sh help

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

_maybe_activate_venv() {
  if [[ -f "$ROOT/venv/bin/activate" ]]; then
    # shellcheck source=/dev/null
    source "$ROOT/venv/bin/activate"
  elif [[ -f "$ROOT/.venv/bin/activate" ]]; then
    # shellcheck source=/dev/null
    source "$ROOT/.venv/bin/activate"
  fi
}

usage() {
  cat <<'USAGE'
Consensus evaluation orchestration (replaces former run_eval_*.sh / run_hybrid_tune_gcp.sh / s.sh).

  ./run_consensus.sh help
  ./run_consensus.sh eval-full
  ./run_consensus.sh eval-parallel-seeded
  ./run_consensus.sh eval-w1-w3-isolated
  ./run_consensus.sh tune-hybrid
  ./run_consensus.sh eval-phased
  ./run_consensus.sh eval-debug-one
  ./run_consensus.sh eval-suite [--mode M] [--w N] [--group-count N] [--git]

eval-full — sequential full run (all windows × biases × modules). Env: CONS_EVAL_WORKERS,
  EVAL_WINDOWS, GROUPS_COUNT, POPULATION_BIASES / POP_BIAS, NDCG_K, MODE, SKIP_DATASET, PYTHON, …

eval-parallel-seeded — isolated workspaces under RUNS_ROOT (parallel jobs + optional seed pass).
  Env: PARALLEL_JOBS, WORKERS_PER_JOB, SEED_FIRST, SEED_WORKERS, GROUP_TYPES, RUNS_ROOT, …

eval-w1-w3-isolated — two rsync workspaces, W=1 and W=3 in parallel (calls eval-full inside each).
  Env: SRC (repo copy source, default ~/analysis), W1_WORKERS, W3_WORKERS, …

tune-hybrid — H0/H1 hyperparameter tuning (validation), not paper eval.
  Env: HYBRID_TUNE_STRATEGY=serial_hybrids|parallel_hybrids, WINDOWS, MERGE_CACHE_BACK, …

eval-phased — two-phase window run (phase1 then all 7 modules on PHASE2_W). Env: PHASE1_W, PHASE2_W.

eval-debug-one — one module + debug profile + NDCG table from latest pickle. Env: MODULE, W, …

eval-suite — run 7 modules once (default W=10) with optional eval_notify.sh logging. Options: --mode, --w, --group-count, --git

sync-gcp — rsync this repo tree to a remote host (personal convenience; override with SYNC_* env).

notify helper (unchanged): ./eval_notify.sh
USAGE
}

# --- eval-full (sequential) ---
cmd_eval_full() {
  _maybe_activate_venv
  export CONS_EVAL_WORKERS="${CONS_EVAL_WORKERS:-6}"
  local -a WINDOWS
  read -r -a WINDOWS <<< "${EVAL_WINDOWS:-1 3 5 10}"
  local GROUPS_COUNT="${GROUPS_COUNT:-1000}"
  local POPULATION_BIASES
  if [[ -n "${POP_BIAS+x}" && -n "${POP_BIAS}" ]]; then
    POPULATION_BIASES="$POP_BIAS"
  else
    POPULATION_BIASES="${POPULATION_BIASES:-0 1 2}"
  fi
  read -r -a BIAS_ARR <<< "$POPULATION_BIASES"
  local NDCG_K="${NDCG_K:-20}"
  local MODE="${MODE:-compute}"
  local PYTHON="${PYTHON:-python3}"
  local BASE_EXTRA=(
    --group-types similar outlier random
  )
  if [[ "${SKIP_DATASET:-0}" == "1" ]]; then
    :
  else
    "$PYTHON" -m evaluation_frameworks.consensus_evaluation.evaluation.evaluation_preparation.eval_dataset_preparation \
      --group-size "${DATASET_GROUP_SIZE:-3}" \
      --min-com "${MIN_COM:-10}"
  fi
  local EVAL_MODULES=(
    evaluation_frameworks.consensus_evaluation.evaluation.evaluations.eval_async_with_sigmoid_policy_simple_priority_individual_rec
    evaluation_frameworks.consensus_evaluation.evaluation.evaluations.eval_async_static_policy_simple_priority_function_individual_rec
    evaluation_frameworks.consensus_evaluation.evaluation.evaluations.eval_async_static_policy_simple_priority_function_group_rec
    evaluation_frameworks.consensus_evaluation.evaluation.evaluations.eval_sync_without_feedback
    evaluation_frameworks.consensus_evaluation.evaluation.evaluations.eval_sync_with_feedback_ema
    evaluation_frameworks.consensus_evaluation.evaluation.evaluations.eval_hybrid_general_rec_individual
    evaluation_frameworks.consensus_evaluation.evaluation.evaluations.eval_hybrid_updatable
  )
  for W in "${WINDOWS[@]}"; do
    local COMMON=(
      --mode "$MODE"
      --window-size "$W"
      --groups-count "$GROUPS_COUNT"
      --population-biases "${BIAS_ARR[@]}"
      --ndcg-k "$NDCG_K"
      "${BASE_EXTRA[@]}"
    )
    for mod in "${EVAL_MODULES[@]}"; do
      "$PYTHON" -m "$mod" "${COMMON[@]}"
    done
  done
}

# --- eval-parallel-seeded ---
cmd_eval_parallel_seeded() {
  _maybe_activate_venv
  local PYTHON="${PYTHON:-python3}"
  local MODE="${MODE:-compute}"
  local NDCG_K="${NDCG_K:-20}"
  local GROUPS_COUNT="${GROUPS_COUNT:-1000}"
  local GROUP_TYPES="${GROUP_TYPES:-similar outlier random}"
  local RUNS_ROOT="${RUNS_ROOT:-$HOME/analysis_runs}"
  local PARALLEL_JOBS="${PARALLEL_JOBS:-4}"
  local WORKERS_PER_JOB="${WORKERS_PER_JOB:-20}"
  local SEED_WORKERS="${SEED_WORKERS:-32}"
  local SEED_FIRST="${SEED_FIRST:-1}"

  local -a WINDOWS
  read -r -a WINDOWS <<< "${EVAL_WINDOWS:-1 3 5 10}"
  local POPULATION_BIASES
  if [[ -n "${POP_BIAS+x}" && -n "${POP_BIAS}" ]]; then
    POPULATION_BIASES="$POP_BIAS"
  else
    POPULATION_BIASES="${POPULATION_BIASES:-0 1 2}"
  fi
  read -r -a BIAS_ARR <<< "$POPULATION_BIASES"
  read -r -a GROUP_TYPE_ARR <<< "$GROUP_TYPES"

  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
  export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
  export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
  export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

  local EVAL_MODULES=(
    evaluation_frameworks.consensus_evaluation.evaluation.evaluations.eval_async_with_sigmoid_policy_simple_priority_individual_rec
    evaluation_frameworks.consensus_evaluation.evaluation.evaluations.eval_async_static_policy_simple_priority_function_individual_rec
    evaluation_frameworks.consensus_evaluation.evaluation.evaluations.eval_async_static_policy_simple_priority_function_group_rec
    evaluation_frameworks.consensus_evaluation.evaluation.evaluations.eval_sync_without_feedback
    evaluation_frameworks.consensus_evaluation.evaluation.evaluations.eval_sync_with_feedback_ema
    evaluation_frameworks.consensus_evaluation.evaluation.evaluations.eval_hybrid_general_rec_individual
    evaluation_frameworks.consensus_evaluation.evaluation.evaluations.eval_hybrid_updatable
  )

  local STAMP BASE_DIR LOG_DIR
  STAMP="$(date +%Y%m%d_%H%M%S)"
  BASE_DIR="$RUNS_ROOT/full_eval_parallel_$STAMP"
  LOG_DIR="$BASE_DIR/logs"
  mkdir -p "$LOG_DIR"

  echo "[parallel-eval] ROOT=$ROOT"
  echo "[parallel-eval] BASE_DIR=$BASE_DIR"
  echo "[parallel-eval] WINDOWS=${WINDOWS[*]}"
  echo "[parallel-eval] BIASES=${BIAS_ARR[*]}"
  echo "[parallel-eval] GROUP_TYPES=${GROUP_TYPE_ARR[*]}"
  echo "[parallel-eval] PARALLEL_JOBS=$PARALLEL_JOBS WORKERS_PER_JOB=$WORKERS_PER_JOB"

  if [[ "${SKIP_DATASET:-0}" != "1" ]]; then
    echo "[parallel-eval] Running dataset preparation once..."
    "$PYTHON" -m evaluation_frameworks.consensus_evaluation.evaluation.evaluation_preparation.eval_dataset_preparation \
      --group-size "${DATASET_GROUP_SIZE:-3}" \
      --min-com "${MIN_COM:-10}"
  fi

  if [[ "$SEED_FIRST" == "1" ]]; then
    echo "[parallel-eval] Seed pass: building shared heavy caches once (single writer)."
    local SEED_W="${WINDOWS[0]}"
    local SEED_B="${BIAS_ARR[0]}"
    local SEED_G="${GROUP_TYPE_ARR[0]}"
    CONS_EVAL_WORKERS="$SEED_WORKERS" \
      "$PYTHON" -m "${EVAL_MODULES[0]}" \
      --mode "$MODE" \
      --window-size "$SEED_W" \
      --groups-count "$GROUPS_COUNT" \
      --population-biases "$SEED_B" \
      --ndcg-k "$NDCG_K" \
      --group-types "$SEED_G" \
      >"$LOG_DIR/seed.log" 2>&1 || true
    echo "[parallel-eval] Seed log: $LOG_DIR/seed.log"
  fi

  start_job() {
    local module="$1"
    local window="$2"
    local module_short="${module##*.}"
    local workdir="$BASE_DIR/work_w${window}_${module_short}"
    local logfile="$LOG_DIR/w${window}_${module_short}.log"

    mkdir -p "$workdir"

    rsync -a --delete \
      --exclude '.git/' \
      --exclude 'venv/' \
      --exclude '.venv/' \
      --exclude '__pycache__/' \
      --exclude 'logs/' \
      --exclude '*.log' \
      "$ROOT/" "$workdir/"

    ln -sfn . "$workdir/analysis"

    (
      cd "$workdir"
      CONS_EVAL_WORKERS="$WORKERS_PER_JOB" \
        "$PYTHON" -m "$module" \
        --mode "$MODE" \
        --window-size "$window" \
        --groups-count "$GROUPS_COUNT" \
        --population-biases "${BIAS_ARR[@]}" \
        --ndcg-k "$NDCG_K" \
        --group-types "${GROUP_TYPE_ARR[@]}"
    ) >"$logfile" 2>&1 &

    echo "[parallel-eval] started: w=$window module=$module_short pid=$! log=$logfile"
  }

  local running=0
  for w in "${WINDOWS[@]}"; do
    for mod in "${EVAL_MODULES[@]}"; do
      start_job "$mod" "$w"
      running=$((running + 1))
      if ((running >= PARALLEL_JOBS)); then
        wait -n
        running=$((running - 1))
      fi
    done
  done

  wait
  echo "[parallel-eval] all jobs finished."
  echo "[parallel-eval] logs: $LOG_DIR"
}

# --- W=1 and W=3 isolated (uses eval-full in each workdir after rsync) ---
cmd_eval_w1_w3_isolated() {
  export OMP_NUM_THREADS=1
  export OPENBLAS_NUM_THREADS=1
  export MKL_NUM_THREADS=1
  export NUMEXPR_NUM_THREADS=1

  local SRC="${SRC:-$HOME/analysis}"
  local RUNS_ROOT="${RUNS_ROOT:-$HOME/analysis_runs}"
  local STAMP
  STAMP="$(date +%Y%m%d_%H%M%S)"
  local W1_WORKERS="${W1_WORKERS:-45}"
  local W3_WORKERS="${W3_WORKERS:-45}"
  local GROUPS_COUNT="${GROUPS_COUNT:-1000}"
  local POPULATION_BIASES="${POPULATION_BIASES:-0 1 2}"

  mkdir -p "$RUNS_ROOT/logs"

  local W1_DIR W3_DIR
  W1_DIR="$RUNS_ROOT/w1_${STAMP}"
  W3_DIR="$RUNS_ROOT/w3_${STAMP}"

  echo "[prep] Creating isolated workspaces..."
  mkdir -p "$W1_DIR" "$W3_DIR"
  rsync -a --delete \
    --exclude '.git/' \
    --exclude 'venv/' \
    --exclude '.venv/' \
    --exclude '__pycache__/' \
    "$SRC/" "$W1_DIR/"
  rsync -a --delete \
    --exclude '.git/' \
    --exclude 'venv/' \
    --exclude '.venv/' \
    --exclude '__pycache__/' \
    "$SRC/" "$W3_DIR/"

  echo "[run] Starting W=1 and W=3 in parallel (isolated caches)..."
  (
    cd "$W1_DIR"
    CONS_EVAL_WORKERS="$W1_WORKERS" GROUPS_COUNT="$GROUPS_COUNT" EVAL_WINDOWS="1" POPULATION_BIASES="$POPULATION_BIASES" RUN_NUM=1001 \
      bash ./run_consensus.sh eval-full
  ) >"$RUNS_ROOT/logs/w1_${STAMP}.log" 2>&1 &
  local PID1=$!

  (
    cd "$W3_DIR"
    CONS_EVAL_WORKERS="$W3_WORKERS" GROUPS_COUNT="$GROUPS_COUNT" EVAL_WINDOWS="3" POPULATION_BIASES="$POPULATION_BIASES" RUN_NUM=1003 \
      bash ./run_consensus.sh eval-full
  ) >"$RUNS_ROOT/logs/w3_${STAMP}.log" 2>&1 &
  local PID2=$!

  wait "$PID1"
  wait "$PID2"

  echo "[done] All jobs finished."
  echo "[done] Logs:"
  echo "  $RUNS_ROOT/logs/w1_${STAMP}.log"
  echo "  $RUNS_ROOT/logs/w3_${STAMP}.log"
}

# --- hybrid tune H0 + H1 ---
cmd_tune_hybrid() {
  _maybe_activate_venv
  local PYTHON="${PYTHON:-python3}"
  local GROUPS_COUNT="${GROUPS_COUNT:-100}"
  local NDCG_K="${NDCG_K:-20}"
  local POPULATION_BIASES="${POPULATION_BIASES:-0}"
  local -a READARRAY_WIN
  read -r -a READARRAY_WIN <<< "${WINDOWS:-1 3 5 10}"

  local HYBRID_TUNE_STRATEGY="${HYBRID_TUNE_STRATEGY:-parallel_hybrids}"
  local MERGE_CACHE_BACK="${MERGE_CACHE_BACK:-1}"

  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
  export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
  export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
  export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
  export CONS_EVAL_USE_PROCESS_POOL="${CONS_EVAL_USE_PROCESS_POOL:-1}"

  _detect_ncores() {
    if [[ -n "${HYBRID_TUNE_NPROC:-}" ]] && [[ "${HYBRID_TUNE_NPROC}" =~ ^[0-9]+$ ]]; then
      echo "$HYBRID_TUNE_NPROC"
      return
    fi
    if command -v nproc >/dev/null 2>&1; then
      nproc
      return
    fi
    if command -v getconf >/dev/null 2>&1; then
      getconf _NPROCESSORS_ONLN 2>/dev/null || echo 8
      return
    fi
    echo 8
  }
  local NCORES
  NCORES="$(_detect_ncores)"

  local GT_DEFAULT="${TUNE_HYBRID_GROUPTYPE_WORKERS:-1}"
  export TUNE_HYBRID_H0_GROUPTYPE_WORKERS="${TUNE_HYBRID_H0_GROUPTYPE_WORKERS:-$GT_DEFAULT}"
  export TUNE_HYBRID_UPD_GROUPTYPE_WORKERS="${TUNE_HYBRID_UPD_GROUPTYPE_WORKERS:-$GT_DEFAULT}"

  if [[ -z "${CONS_EVAL_WORKERS:-}" ]]; then
    local GT_USE="$TUNE_HYBRID_H0_GROUPTYPE_WORKERS"
    if [[ "$TUNE_HYBRID_UPD_GROUPTYPE_WORKERS" -gt "$GT_USE" ]]; then
      GT_USE="$TUNE_HYBRID_UPD_GROUPTYPE_WORKERS"
    fi
    local SLOT
    if [[ "$HYBRID_TUNE_STRATEGY" == "parallel_hybrids" ]]; then
      SLOT=$((NCORES / 2))
    else
      SLOT="$NCORES"
    fi
    export CONS_EVAL_WORKERS=$((SLOT / GT_USE))
    if [[ "$CONS_EVAL_WORKERS" -lt 1 ]]; then
      export CONS_EVAL_WORKERS=1
    fi
  else
    export CONS_EVAL_WORKERS
  fi

  local RUNS_ROOT="${RUNS_ROOT:-$HOME/analysis_runs}"
  local LOG_DIR="${LOG_DIR:-$RUNS_ROOT/hybrid_tune_logs}"
  mkdir -p "$LOG_DIR"

  local STAMP
  STAMP="$(date +%Y%m%d_%H%M%S)"
  local BASE_DIR="${HYBRID_TUNE_BASE:-$RUNS_ROOT/hybrid_tune_$STAMP}"

  local MOD_H0="evaluation_frameworks.consensus_evaluation.evaluation.evaluations.tune_hybrid_all_params"
  local MOD_H1="evaluation_frameworks.consensus_evaluation.evaluation.evaluations.tune_hybrid_individual_updatable"

  echo "[hybrid-tune] TUNE (validation), ne eval — ROOT=$ROOT"
  echo "[hybrid-tune] strategy=$HYBRID_TUNE_STRATEGY  detected_cores=$NCORES (override: HYBRID_TUNE_NPROC)  WINDOWS=${READARRAY_WIN[*]}"
  echo "[hybrid-tune] GROUPS_COUNT=$GROUPS_COUNT  CONS_EVAL_WORKERS=$CONS_EVAL_WORKERS"
  echo "[hybrid-tune] TUNE_HYBRID_H0_GROUPTYPE_WORKERS=$TUNE_HYBRID_H0_GROUPTYPE_WORKERS  TUNE_HYBRID_UPD_GROUPTYPE_WORKERS=$TUNE_HYBRID_UPD_GROUPTYPE_WORKERS"
  echo "[hybrid-tune] logs -> $LOG_DIR"
  if [[ "$HYBRID_TUNE_STRATEGY" == "parallel_hybrids" ]]; then
    echo "[hybrid-tune] parallel → izolované workdiry pod $BASE_DIR (rsync jako parallel_seeded)"
    echo "[hybrid-tune] MERGE_CACHE_BACK=$MERGE_CACHE_BACK → po W sloučení do $ROOT/cache/"
    if [[ "${HYBRID_TUNE_RSYNC_EXCLUDE_CACHE:-1}" != "0" ]]; then
      echo "[hybrid-tune] rsync exclude cache/=yes (šetří disk; tuner si znovu vytvoří pickle v workdir/cache → pak merge)"
    else
      echo "[hybrid-tune] rsync exclude cache/=no — kopíruje se celý ROOT/cache (potřeba hodně místa)"
    fi
    df -h "$ROOT" 2>/dev/null | head -n 4 || true
  fi

  if [[ "${SKIP_DATASET:-0}" != "1" ]]; then
    echo "[hybrid-tune] Dataset preparation (jeden zapisovatel, ROOT)..."
    "$PYTHON" -m evaluation_frameworks.consensus_evaluation.evaluation.evaluation_preparation.eval_dataset_preparation \
      --group-size "${DATASET_GROUP_SIZE:-3}" \
      --min-com "${MIN_COM:-10}"
  fi

  prep_isolated_workdir() {
    local dst="$1"
    mkdir -p "$dst"
    local excl_cache=()
    if [[ "${HYBRID_TUNE_STRATEGY:-}" == "parallel_hybrids" ]] && [[ "${HYBRID_TUNE_RSYNC_EXCLUDE_CACHE:-1}" != "0" ]]; then
      excl_cache+=(--exclude "cache/")
    fi
    rsync -a --delete \
      --exclude '.git/' \
      --exclude 'venv/' \
      --exclude '.venv/' \
      --exclude '__pycache__/' \
      --exclude 'logs/' \
      --exclude '*.log' \
      "${excl_cache[@]}" \
      "$ROOT/" "$dst/"
    mkdir -p "$dst/cache"
    ln -sfn . "$dst/analysis"
  }

  _merge_workdir_cache_to_root() {
    local src="$1"
    if [[ -d "$src/cache" ]]; then
      mkdir -p "$ROOT/cache"
      rsync -a "$src/cache/" "$ROOT/cache/"
      echo "[hybrid-tune] merged $src/cache/ → $ROOT/cache/"
    fi
  }

  _run_mod_in() {
    local workdir="$1"
    local mod="$2"
    local w="$3"
    local log="$4"
    (
      cd "$workdir"
      "$PYTHON" -m "$mod" \
        --mode compute \
        --window-size "$w" \
        --groups-count "$GROUPS_COUNT" \
        --population-biases $POPULATION_BIASES \
        --ndcg-k "$NDCG_K" \
        --save-mode upsert
    ) >"$log" 2>&1
  }

  run_window_serial() {
    local w="$1"
    local stamp
    stamp="$(date +%Y%m%d_%H%M%S)"
    local log0="$LOG_DIR/h0_tune_w${w}_${stamp}.log"
    local log1="$LOG_DIR/h1_tune_w${w}_${stamp}.log"
    echo "[hybrid-tune] W=$w serial: H0 → H1 v ROOT"
    _run_mod_in "$ROOT" "$MOD_H0" "$w" "$log0"
    echo "[hybrid-tune] W=$w H0 done -> $log0"
    _run_mod_in "$ROOT" "$MOD_H1" "$w" "$log1"
    echo "[hybrid-tune] W=$w H1 done -> $log1"
  }

  run_window_parallel_isolated() {
    local w="$1"
    local stamp
    stamp="$(date +%Y%m%d_%H%M%S)"
    local wd0="$BASE_DIR/work_w${w}_h0_tune_hybrid_all_params"
    local wd1="$BASE_DIR/work_w${w}_h1_tune_hybrid_individual_updatable"
    local log0="$LOG_DIR/h0_tune_w${w}_${stamp}.log"
    local log1="$LOG_DIR/h1_tune_w${w}_${stamp}.log"

    mkdir -p "$BASE_DIR"
    echo "[hybrid-tune] W=$w příprava izolovaných workdirů..."
    prep_isolated_workdir "$wd0"
    prep_isolated_workdir "$wd1"

    echo "[hybrid-tune] W=$w parallel: H0 + H1 (oddělené cache/)"
    _run_mod_in "$wd0" "$MOD_H0" "$w" "$log0" &
    local pid0=$!
    _run_mod_in "$wd1" "$MOD_H1" "$w" "$log1" &
    local pid1=$!
    wait "$pid0" "$pid1"
    echo "[hybrid-tune] W=$w hotovo -> $log0 $log1"

    if [[ "$MERGE_CACHE_BACK" == "1" ]]; then
      _merge_workdir_cache_to_root "$wd0"
      _merge_workdir_cache_to_root "$wd1"
    else
      echo "[hybrid-tune] merge přeskočen; výsledky cache jsou jen v $wd0 a $wd1"
    fi
  }

  local w
  for w in "${READARRAY_WIN[@]}"; do
    if [[ "$HYBRID_TUNE_STRATEGY" == "parallel_hybrids" ]]; then
      run_window_parallel_isolated "$w"
    else
      run_window_serial "$w"
    fi
  done

  echo "[hybrid-tune] hotovo."
}

# --- phased two-window pipeline ---
cmd_eval_phased() {
  _maybe_activate_venv
  local PYTHON="${PYTHON:-python3}"
  local MODE="${MODE:-compute}"
  local GROUPS_COUNT="${GROUPS_COUNT:-100}"
  local PHASE1_W="${PHASE1_W:-5}"
  local PHASE2_W="${PHASE2_W:-10}"
  local POPULATION_BIASES="${POPULATION_BIASES:-0}"

  local PHASE1_MODULES=(
    evaluation_frameworks.consensus_evaluation.evaluation.evaluations.eval_async_static_policy_simple_priority_function_group_rec
    evaluation_frameworks.consensus_evaluation.evaluation.evaluations.eval_async_static_policy_simple_priority_function_individual_rec
    evaluation_frameworks.consensus_evaluation.evaluation.evaluations.eval_async_with_sigmoid_policy_simple_priority_individual_rec
    evaluation_frameworks.consensus_evaluation.evaluation.evaluations.eval_sync_without_feedback
    evaluation_frameworks.consensus_evaluation.evaluation.evaluations.eval_sync_with_feedback_ema
  )

  local PHASE2_MODULES=(
    evaluation_frameworks.consensus_evaluation.evaluation.evaluations.eval_async_with_sigmoid_policy_simple_priority_individual_rec
    evaluation_frameworks.consensus_evaluation.evaluation.evaluations.eval_async_static_policy_simple_priority_function_individual_rec
    evaluation_frameworks.consensus_evaluation.evaluation.evaluations.eval_async_static_policy_simple_priority_function_group_rec
    evaluation_frameworks.consensus_evaluation.evaluation.evaluations.eval_sync_without_feedback
    evaluation_frameworks.consensus_evaluation.evaluation.evaluations.eval_sync_with_feedback_ema
    evaluation_frameworks.consensus_evaluation.evaluation.evaluations.eval_hybrid_general_rec_individual
    evaluation_frameworks.consensus_evaluation.evaluation.evaluations.eval_hybrid_updatable
  )

  run_mod_phase1() {
    local mod="$1"
    echo "###########################################################################"
    echo "###  Phase 1 (A0–A2, S0–S1)  W=${PHASE1_W}  groups=${GROUPS_COUNT}  biases=${POPULATION_BIASES}  ###"
    echo "###  $mod"
    echo "###########################################################################"
    "$PYTHON" -m "$mod" \
      --mode "$MODE" \
      --window-size "$PHASE1_W" \
      --groups-count "$GROUPS_COUNT" \
      --population-biases $POPULATION_BIASES \
      --group-types similar outlier random
  }

  run_mod_phase2() {
    local mod="$1"
    echo "###########################################################################"
    echo "###  Phase 2 (all 7)  W=${PHASE2_W}  groups=${GROUPS_COUNT}  biases=${POPULATION_BIASES}  ###"
    echo "###  $mod"
    echo "###########################################################################"
    "$PYTHON" -m "$mod" \
      --mode "$MODE" \
      --window-size "$PHASE2_W" \
      --groups-count "$GROUPS_COUNT" \
      --population-biases $POPULATION_BIASES
  }

  echo "== Phase 1: ${#PHASE1_MODULES[@]} modulů, W=${PHASE1_W}, group types = similar outlier random, groups-count=${GROUPS_COUNT}, population-biases=${POPULATION_BIASES} =="
  local mod
  for mod in "${PHASE1_MODULES[@]}"; do
    run_mod_phase1 "$mod"
  done

  echo ""
  echo "== Phase 1 hotová. Spouštím Phase 2: ${#PHASE2_MODULES[@]} modulů, W=${PHASE2_W}, výchozí group types, groups-count=${GROUPS_COUNT}, population-biases=${POPULATION_BIASES} =="

  for mod in "${PHASE2_MODULES[@]}"; do
    run_mod_phase2 "$mod"
  done

  echo ""
  echo "== Obě fáze hotové."
  echo "    Cache: cache/cons_evaluations/w_${PHASE1_W}/… (fáze 1), w_${PHASE2_W}/… (fáze 2)"
}

# --- single-module debug ---
cmd_eval_debug_one() {
  _maybe_activate_venv
  local PYTHON="${PYTHON:-python3}"
  local MODULE="${MODULE:-evaluation_frameworks.consensus_evaluation.evaluation.evaluations.eval_async_static_policy_simple_priority_function_individual_rec}"
  local EVAL_NAME="${EVAL_NAME:-eval_async_static_policy_simple_priority_function_individual_rec.py}"
  local W="${W:-10}"
  local GROUPS_COUNT="${GROUPS_COUNT:-100}"
  local POP_BIAS="${POP_BIAS:-0}"
  local NDCG_K="${NDCG_K:-20}"
  local GROUP_TYPES="${GROUP_TYPES:-random}"
  local MODE="${MODE:-compute}"
  local SAVE_MODE="${SAVE_MODE:-append}"
  local WORKERS="${CONS_EVAL_WORKERS:-6}"

  export CONS_EVAL_WORKERS="$WORKERS"
  export PYTHONPATH="${PYTHONPATH:-.}"

  local OLD_CONTEXT_BUG=0
  local _b="${CONS_EVAL_REINTRODUCE_OLD_CONTEXT_GROUPS_BUG:-}"
  if [[ "${_b,,}" == "1" || "${_b,,}" == "true" || "${_b,,}" == "yes" ]]; then
    export CONS_EVAL_REINTRODUCE_OLD_CONTEXT_GROUPS_BUG
    OLD_CONTEXT_BUG=1
    echo "!! CONS_EVAL_REINTRODUCE_OLD_CONTEXT_GROUPS_BUG=$_b (old build_context_groups_data hardcode active)"
  fi
  unset _b

  echo "== Run one evaluation =="
  echo "module=$MODULE"
  echo "workers=$CONS_EVAL_WORKERS, W=$W, groups_count=$GROUPS_COUNT, ndcg_k=$NDCG_K, group_types=[$GROUP_TYPES]"
  echo "------------------------------------------------------------------------"
  echo " POPULATION MOOD BIAS → předáno jako --population-biases: $POP_BIAS"
  echo " (více hodnot v experimentu = více bloků >>> Bias k/n v Python výstupu)"
  echo "------------------------------------------------------------------------"

  local PROFILE_TAG="workers_${CONS_EVAL_WORKERS}"
  if [[ "$OLD_CONTEXT_BUG" -eq 1 ]]; then
    PROFILE_TAG="${PROFILE_TAG}_old_context_bug"
  fi

  "$PYTHON" -m "$MODULE" \
    --mode "$MODE" \
    --window-size "$W" \
    --groups-count "$GROUPS_COUNT" \
    --population-biases "$POP_BIAS" \
    --ndcg-k "$NDCG_K" \
    --group-types $GROUP_TYPES \
    --save-mode "$SAVE_MODE" \
    --debug-profile \
    --debug-profile-tag "$PROFILE_TAG"

  echo ""
  echo "== Resolve latest pickle path =="
  local LATEST_PKL
  LATEST_PKL="$(
    W="$W" EVAL_NAME="$EVAL_NAME" GROUPS_COUNT="$GROUPS_COUNT" "$PYTHON" - <<'PY'
import os
from pathlib import Path
from evaluation_frameworks.consensus_evaluation.evaluation.evaluations.config import evaluation_results_dir

w = os.environ["W"]
eval_type = "test"
evaluation_name = os.environ["EVAL_NAME"]
groups_count = int(os.environ["GROUPS_COUNT"])

base = evaluation_results_dir(
    window_size=w,
    eval_type=eval_type,
    evaluation_name=evaluation_name,
    groups_count=groups_count,
    layout="labeled",
)

pickles = list(base.glob("*.pkl"))
if not pickles:
    raise SystemExit(f"No pickle found in: {base}")

latest = max(pickles, key=lambda p: p.stat().st_mtime)
print(str(latest))
PY
  )"

  echo "latest_pickle=$LATEST_PKL"
  echo ""
  echo "== NDCG table =="
  "$PYTHON" inspect_ndcg_pickle.py --pickle "$LATEST_PKL" --k "$NDCG_K" --bias "$POP_BIAS"
}

# --- sequential suite + notify (legacy run_eval.sh) ---
cmd_eval_suite() {
  set +e
  set +u
  local NOTIFY="$ROOT/eval_notify.sh"
  local DO_GIT=true
  local MODE="compute"
  local W_SIZE=10
  local GROUPS_COUNT="1000"

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --mode)
        MODE="${2:-auto}"
        shift 2
        ;;
      --w)
        W_SIZE="${2:-auto}"
        shift 2
        ;;
      --group-count)
        GROUPS_COUNT="${2:-}"
        shift 2
        ;;
      --git | -g)
        DO_GIT=true
        shift
        ;;
      *)
        echo "Unknown argument: $1" >&2
        exit 2
        ;;
    esac
  done

  if [[ ! -x "$NOTIFY" ]]; then
    echo "Warning: $NOTIFY not executable." >&2
  fi

  local JOBS=(
    "eval_hybrid_general_rec_individual evaluation_frameworks.consensus_evaluation.evaluation.evaluations.eval_hybrid_general_rec_individual"
    "eval_hybrid_updatable evaluation_frameworks.consensus_evaluation.evaluation.evaluations.eval_hybrid_updatable"
    "eval_sync_without_feedback evaluation_frameworks.consensus_evaluation.evaluation.evaluations.eval_sync_without_feedback"
    "eval_sync_with_feedback_ema evaluation_frameworks.consensus_evaluation.evaluation.evaluations.eval_sync_with_feedback_ema"
    "eval_async_static_policy_simple_priority_function_individual_rec evaluation_frameworks.consensus_evaluation.evaluation.evaluations.eval_async_static_policy_simple_priority_function_individual_rec"
    "eval_async_with_sigmoid_policy_simple_priority_individual_rec evaluation_frameworks.consensus_evaluation.evaluation.evaluations.eval_async_with_sigmoid_policy_simple_priority_individual_rec"
    "eval_async_static_policy_simple_priority_function_group_rec evaluation_frameworks.consensus_evaluation.evaluation.evaluations.eval_async_static_policy_simple_priority_function_group_rec"
  )

  iso_now() { date -Iseconds; }
  epoch_now() { date +%s; }

  run_one() {
    local label="$1"
    local module="$2"
    local start_iso end_iso start_s end_s dur_s status code note

    echo "────────────────────────────────────────────────────────"
    echo "▶ Executing: $label  (module: $module, --mode $MODE)"
    start_iso="$(iso_now)"
    start_s="$(epoch_now)"

    local PY_ARGS=(--mode "$MODE" --window-size "$W_SIZE")
    if [[ -n "$GROUPS_COUNT" ]]; then
      PY_ARGS+=(--groups-count "$GROUPS_COUNT")
    fi

    python3 -m "$module" "${PY_ARGS[@]}"
    code=$?

    end_iso="$(iso_now)"
    end_s="$(epoch_now)"
    dur_s=$((end_s - start_s))

    if [[ $code -eq 0 ]]; then
      status="success"
      note="$label finished ok (mode=$MODE)"
    else
      status="fail"
      note="$label failed (mode=$MODE)"
    fi

    if $DO_GIT; then
      "$NOTIFY" "$label" "$status" -c "$code" -s "$start_iso" -e "$end_iso" -d "$dur_s" -n "$note" -g
    else
      "$NOTIFY" "$label" "$status" -c "$code" -s "$start_iso" -e "$end_iso" -d "$dur_s" -n "$note"
    fi

    echo "⏱  duration: ${dur_s}s | exit code: $code | status: $status"
    return 0
  }

  local overall_ok=true
  local pair
  for pair in "${JOBS[@]}"; do
    local label="${pair%% *}"
    local module="${pair#* }"
    run_one "$label" "$module" || overall_ok=false
  done

  echo "────────────────────────────────────────────────────────"
  if $overall_ok; then
    echo "✅ Everything completed."
  else
    echo "⚠️  Some runs failed."
  fi
}

cmd_sync_gcp() {
  local SYNC_SRC="${SYNC_SRC:-$HOME/projects/thesis/analysis}"
  local SYNC_REMOTE="${SYNC_REMOTE:-cross-bit@34.12.58.184:~/analysis/}"
  local SYNC_SSH_KEY="${SYNC_SSH_KEY:-$HOME/.ssh/google_compute_engine}"
  rsync -azP --delete \
    -e "ssh -i $SYNC_SSH_KEY" \
    --exclude 'cache/' \
    --exclude 'cache/**' \
    --exclude '.git/' \
    --exclude '__pycache__/' \
    --exclude 'venv/' \
    --exclude '.venv/' \
    --exclude '*.pyc' \
    --exclude '.pytest_cache/' \
    --exclude '.mypy_cache/' \
    --exclude '.idea/' \
    --exclude '.vscode/' \
    --exclude 'results/' \
    --exclude 'logs/' \
    --exclude '*.log' \
    --exclude 'restaurant_data/**' \
    --exclude '*.pkl' \
    "$SYNC_SRC/" \
    "$SYNC_REMOTE"
}

main() {
  local cmd="${1:-help}"
  shift || true
  case "$cmd" in
    help | -h | --help) usage ;;
    eval-full) cmd_eval_full "$@" ;;
    eval-parallel-seeded) cmd_eval_parallel_seeded "$@" ;;
    eval-w1-w3-isolated) cmd_eval_w1_w3_isolated "$@" ;;
    tune-hybrid) cmd_tune_hybrid "$@" ;;
    eval-phased) cmd_eval_phased "$@" ;;
    eval-debug-one) cmd_eval_debug_one "$@" ;;
    eval-suite) cmd_eval_suite "$@" ;;
    sync-gcp) cmd_sync_gcp "$@" ;;
    *)
      echo "Unknown command: $cmd" >&2
      usage >&2
      exit 2
      ;;
  esac
}

main "$@"
