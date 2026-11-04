#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${WORKFLOW_PYTHON:-python3}" "$repo_dir/scripts/workflow.py" "$@"
