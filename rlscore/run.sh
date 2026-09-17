#!/usr/bin/env bash
# Convenience wrapper around the three commands needed to set up, train, and
# serve the RL scheduler (see pkg/simulator/plugin/rl_score.go for the Go
# side that dials into `serve`).
set -euo pipefail
cd "$(dirname "$0")/.."

VENV_DIR="${RLSCORE_VENV:-rlscore/.venv}"

usage() {
  cat <<USAGE
Usage: rlscore/run.sh <command> [args...]

Commands:
  setup                   Create a venv at \$RLSCORE_VENV (default: $VENV_DIR)
                           and install rlscore/requirements.txt into it.

  train [train args...]   Run PPO training (python -m rlscore.train).
                           Defaults to the openb_node_list_gpu_node /
                           openb_pod_list_gpushare60 traces, writing
                           rlscore/checkpoint.pt. Any args given override the
                           defaults, e.g.:
                             rlscore/run.sh train --iterations 500
                             rlscore/run.sh train --pods data/csv/openb_pod_list_multigpu30.csv

  serve [serve args...]   Run the gRPC policy server (python -m rlscore.server)
                           against rlscore/checkpoint.pt on port 50051.
                           This is what pluginConfig.args.inferenceAddr in
                           example/rl-scheduler-config.yaml should point at.

Environment:
  RLSCORE_VENV             Where 'setup' creates the venv, and where the
                           other commands look for it (default: rlscore/.venv).
USAGE
}

activate() {
  if [ ! -d "$VENV_DIR" ]; then
    echo "No venv found at $VENV_DIR -- run 'rlscore/run.sh setup' first." >&2
    exit 1
  fi
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
}

cmd="${1:-}"
[ $# -gt 0 ] && shift

case "$cmd" in
  setup)
    python3 -m venv "$VENV_DIR"
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
    pip install --upgrade pip
    pip install -r rlscore/requirements.txt
    echo "Ready. Activate with: source $VENV_DIR/bin/activate"
    ;;
  train)
    activate
    python -m rlscore.train \
      --nodes data/csv/openb_node_list_gpu_node.csv \
      --pods data/csv/openb_pod_list_gpushare60.csv \
      --iterations 200 \
      --out rlscore/checkpoint.pt \
      "$@"
    ;;
  serve)
    activate
    python -m rlscore.server --checkpoint rlscore/checkpoint.pt "$@"
    ;;
  -h|--help|help|"")
    usage
    ;;
  *)
    echo "Unknown command: $cmd" >&2
    usage
    exit 1
    ;;
esac
