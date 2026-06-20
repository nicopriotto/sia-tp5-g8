#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

MODE="${1:-formal}"
if [[ "$MODE" != "formal" && "$MODE" != "quick" ]]; then
  echo "Uso: bash experiments/vae/run_all.sh [formal|quick]"
  exit 1
fi

LOG_DIR="$ROOT_DIR/experiments/output/vae/logs"
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

run_experiment "beta" "experiments/vae/beta.py"
run_experiment "latent_dim" "experiments/vae/latent_dim.py"

log "Run completo finalizado mode=${MODE}"
