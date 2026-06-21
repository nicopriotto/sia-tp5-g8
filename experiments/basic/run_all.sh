#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

MODE="${1:-formal}"
if [[ "$MODE" != "formal" && "$MODE" != "quick" ]]; then
  echo "Uso: bash experiments/basic/run_all.sh [formal|quick]"
  exit 1
fi

LOG_DIR="$ROOT_DIR/experiments/output/basic/logs"
mkdir -p "$LOG_DIR"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/run_all_${MODE}_${TIMESTAMP}.log"

log() {
  local message="$1"
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$message" | tee -a "$LOG_FILE"
}

run_experiment() {
  local experiment_name="$1"
  local script_path="$2"

  log "Inicio experimento=${experiment_name} mode=${MODE}"
  python3 "$script_path" --mode "$MODE" 2>&1 | tee -a "$LOG_FILE"
  log "Fin experimento=${experiment_name} mode=${MODE}"
}

log "Run completo iniciado mode=${MODE}"
log "Log file: ${LOG_FILE}"

run_experiment "loss" "experiments/basic/loss.py"
run_experiment "architecture" "experiments/basic/architecture.py"
run_experiment "activation"   "experiments/basic/activation.py"
run_experiment "optimizer_lr" "experiments/basic/optimizer_lr.py"
run_experiment "regularization_threshold" "experiments/basic/regularization_threshold.py"

log "Run completo finalizado mode=${MODE}"
