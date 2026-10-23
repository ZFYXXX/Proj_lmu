#!/usr/bin/env bash
set -uo pipefail
if [[ $# -lt 1 ]]; then
  echo "Usage: $0 A|B|C|E|H [launch options...]" >&2
  exit 2
fi
experiment="$1"
shift
case "$experiment" in A|B|C|E|H) ;; *) echo "请选择 A/B/C/E/H" >&2; exit 2 ;; esac
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "$script_dir/../../.." && pwd)"
export ANALYSIS_RUN_DIR="${ANALYSIS_RUN_DIR:-$project_root/runs/analysis/audio_redundancy/${experiment}_4gpu_$(date +%Y%m%d_%H%M%S)}"
for argument in "$@"; do
  if [[ "$argument" == "--plan-only" ]]; then
    exec bash "$script_dir/run_mmsufinal_4gpu.sh" "$@" --experiments "$experiment"
  fi
done

log_path="${ANALYSIS_RUN_DIR}.launch.log"
archive_path="${ANALYSIS_RUN_DIR}.portable.tar.gz"
mkdir -p "$(dirname -- "$ANALYSIS_RUN_DIR")"
# The Python supervisor isolates stages for every experiment and for --experiments all.
bash "$script_dir/run_mmsufinal_4gpu.sh" "$@" --experiments "$experiment" 2>&1 | tee -a "$log_path"
statuses=("${PIPESTATUS[@]}")
run_status="${statuses[0]}"
log_status="${statuses[1]}"
if (( run_status == 0 && log_status == 0 )) && [[ -f "$ANALYSIS_RUN_DIR/portable/manifest.json" ]]; then
  if tar -czf "$archive_path" -C "$ANALYSIS_RUN_DIR" portable; then
    echo "实验 $experiment 完成；报告：$ANALYSIS_RUN_DIR/report"
    echo "精简分析包：$archive_path"
    echo "运行日志：$log_path"
    true
  else
    echo "实验 $experiment 已完成，但打包失败；可复制 $ANALYSIS_RUN_DIR/portable" >&2
    false
  fi
else
  echo "实验 $experiment 未完成；查看日志：$log_path" >&2
  false
fi
