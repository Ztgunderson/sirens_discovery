#!/usr/bin/env bash
# Overnight SHADE-Arena discovery generation harness.
# Loops over ALL task pairs (discovery + held-out) x {benign, rogue}, running
# task_completion.py in FULL (non-test) mode against the local vLLM model.
#
# IDEMPOTENT: counts transcripts already present for each (task,mode) and only
# tops up to N. Ordered smallest-env-tasks first. Never launches on its own.
#
# Usage:
#   ./run_generation.sh [N] [--max-minutes M]
#   N          number of experiments per (task,mode)   (default 5, or env N)
#   --max-minutes M   soft wall-clock budget in minutes (default unlimited, or env MAX_MINUTES)
#
# Env overrides: N, MAX_MINUTES, OPENAI_BASE_URL, OPENAI_API_KEY
set -uo pipefail

# --- resolve paths relative to this script (durable copy) ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SHADE_DIR="$SCRIPT_DIR/shade"
PY="$SHADE_DIR/.venv/bin/python"
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

# --- args ---
N="${N:-5}"
MAX_MINUTES="${MAX_MINUTES:-0}"   # 0 = unlimited
while [ $# -gt 0 ]; do
  case "$1" in
    --max-minutes) MAX_MINUTES="$2"; shift 2 ;;
    --max-minutes=*) MAX_MINUTES="${1#*=}"; shift ;;
    [0-9]*) N="$1"; shift ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

# --- model routing (local vLLM, OpenAI-compatible) ---
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://localhost:8000/v1}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"

# --- tasks, ordered smallest-env first (spam & AA first) ---
TASKS=(
  spam_filter_update
  american_airlines_2
  bank_transfer_2
  expenditure_accounting
  api_key_calendar_agendas_2
)

# mode -> variant transcript dir (must match utils/experiment_runner.py log_dir)
#   benign: --include_secondary_task False       -> transcripts_no_secondary_task
#   rogue : secondary task + hint, keep thinking -> transcripts_no_thinking_time
declare -A VARIANT=(
  [benign]="transcripts_no_secondary_task"
  [rogue]="transcripts_no_thinking_time"
)
MODES=(benign rogue)

# count completed transcript_*.json for a (task,mode)
count_done() {
  local task="$1" mode="$2"
  local d="$SHADE_DIR/task_pairs/$task/${VARIANT[$mode]}"
  [ -d "$d" ] || { echo 0; return; }
  find "$d" -type f -name 'transcript_*.json' 2>/dev/null | wc -l
}

run_one() {
  local task="$1" mode="$2" need="$3"
  local common=(--num_experiments "$need" --task_folder "$task"
                --max_concurrent_threads 2 --test_mode False --run_all_tasks False)
  local flags
  if [ "$mode" = "benign" ]; then
    flags=(--include_secondary_task False --give_hint False --remove_thinking_tags False)
  else
    flags=(--include_secondary_task True  --give_hint True  --remove_thinking_tags False)
  fi
  local ts; ts="$(date +%Y%m%d_%H%M%S)"
  local logf="$LOG_DIR/${task}_${mode}_${ts}.log"
  echo "[$(date +%H:%M:%S)] RUN task=$task mode=$mode need=$need -> $logf"
  ( cd "$SHADE_DIR" && "$PY" task_completion.py "${common[@]}" "${flags[@]}" ) >>"$logf" 2>&1
  local rc=$?
  echo "[$(date +%H:%M:%S)] DONE task=$task mode=$mode rc=$rc now_have=$(count_done "$task" "$mode")"
  return $rc
}

START=$SECONDS
budget_exceeded() {
  [ "$MAX_MINUTES" = "0" ] && return 1
  local elapsed=$(( (SECONDS - START) / 60 ))
  [ "$elapsed" -ge "$MAX_MINUTES" ]
}

echo "=== SHADE overnight generation ==="
echo "N=$N  MAX_MINUTES=$MAX_MINUTES  BASE_URL=$OPENAI_BASE_URL"
echo "SHADE_DIR=$SHADE_DIR"
echo "Tasks: ${TASKS[*]}"
echo "Modes: ${MODES[*]}"
echo

for task in "${TASKS[@]}"; do
  for mode in "${MODES[@]}"; do
    if budget_exceeded; then
      echo "[$(date +%H:%M:%S)] Wall-clock budget ($MAX_MINUTES min) reached; stopping."
      echo "=== generation stopped early (budget) ==="
      exit 0
    fi
    have=$(count_done "$task" "$mode")
    need=$(( N - have ))
    if [ "$need" -le 0 ]; then
      echo "[$(date +%H:%M:%S)] SKIP task=$task mode=$mode (have=$have >= N=$N)"
      continue
    fi
    run_one "$task" "$mode" "$need"
  done
done

echo
echo "=== generation complete ==="
for task in "${TASKS[@]}"; do
  for mode in "${MODES[@]}"; do
    echo "  $task/$mode: $(count_done "$task" "$mode")/$N"
  done
done
