#!/usr/bin/env bash
set -euo pipefail

# Run only 50% audio-token retention, reusing the existing five keep50 YAMLs.
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${script_dir}/common.bash"
run_projector_ratio_sweep unisamp_keep50 "$@"
