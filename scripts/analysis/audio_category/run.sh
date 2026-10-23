#!/usr/bin/env bash
set -euo pipefail

category_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "$category_dir/../../.." && pwd)"
export PYTHONPATH="$project_root:$project_root/third_party/frameworks/lmms-eval${PYTHONPATH:+:$PYTHONPATH}"
cd -- "$project_root"
exec "${PYTHON_BIN:-python}" -m scripts.analysis.audio_category.launch "$@"
