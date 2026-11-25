#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

stage=${1:-${ROLE3_STAGE:-all}}
case "$stage" in
  train) bash scripts/role3_sensitivity_train.sh ;;
  eval) bash scripts/role3_sensitivity_eval.sh ;;
  all)
    bash scripts/role3_sensitivity_train.sh
    bash scripts/role3_sensitivity_eval.sh
    ;;
  *)
    echo "Usage: $0 [train|eval|all]" >&2
    exit 2
    ;;
esac
