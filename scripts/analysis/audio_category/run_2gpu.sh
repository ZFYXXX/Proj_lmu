#!/usr/bin/env bash
set -euo pipefail

category_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$category_dir/run.sh" --gpus 0,1 "$@"
