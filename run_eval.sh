#!/usr/bin/env bash
set -uo pipefail

# ---- Konfigurace ----
NOTIFY="./eval_notify.sh"
USE_NOTIFY=false
DO_GIT=true                 # když --git, předá se -g do notify
MODE="compute"              # defaultní --mode
W_SIZE=10
GROUPS_COUNT="1000"               # volitelné

# ---- Parse args ----
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
    --git|-g)
      DO_GIT=true
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done


if [[ -x "$NOTIFY" ]]; then
  USE_NOTIFY=true
else
  echo "Info: notify helper not found, running without notifications." >&2
fi

JOBS=(
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

  # Sestavení argumentů pro python
  PY_ARGS=(--mode "$MODE" --window-size "$W_SIZE")
  if [[ -n "$GROUPS_COUNT" ]]; then
    PY_ARGS+=(--group-eval-size "$GROUPS_COUNT")
  fi

  # Spuštění modulu
  python3 -m "$module" "${PY_ARGS[@]}"
  code=$?

  end_iso="$(iso_now)"
  end_s="$(epoch_now)"
  dur_s=$(( end_s - start_s ))

  if [[ $code -eq 0 ]]; then
    status="success"
    note="$label finished ok (mode=$MODE)"
  else
    status="fail"
    note="$label failed (mode=$MODE)"
  fi

  # Notifications
  if $USE_NOTIFY; then
    if $DO_GIT; then
      "$NOTIFY" "$label" "$status" -c "$code" -s "$start_iso" -e "$end_iso" -d "$dur_s" -n "$note" -g
    else
      "$NOTIFY" "$label" "$status" -c "$code" -s "$start_iso" -e "$end_iso" -d "$dur_s" -n "$note"
    fi
  fi

  echo "⏱  duration: ${dur_s}s | exit code: $code | status: $status"
  return 0
}

# ---- Main loop ----
overall_ok=true
for pair in "${JOBS[@]}"; do
  label="${pair%% *}"
  module="${pair#* }"
  run_one "$label" "$module" || overall_ok=false
done

echo "────────────────────────────────────────────────────────"
if $overall_ok; then
  echo "✅ Everything completed."
else
  echo "⚠️  Some runs failed."
fi