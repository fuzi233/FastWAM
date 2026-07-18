#!/usr/bin/env bash
set -euo pipefail

TOOLS_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=env.sh
source "$TOOLS_DIR/env.sh"

cd "$FASTWAM_REPO"
bash "$TOOLS_DIR/check_assets.sh"

IFS=, read -r -a inference_gpu_ids <<<"$FASTWAM_GPU_LIST"
if [[ ${#inference_gpu_ids[@]} -ne 1 ]]; then
    printf 'ERROR: inference smoke run requires exactly one physical GPU, got %s\n' "$FASTWAM_GPU_LIST" >&2
    exit 1
fi

artifacts_root=${FASTWAM_ARTIFACTS_OVERRIDE:-$FASTWAM_ARTIFACTS}
eval_python=${FASTWAM_EVAL_PYTHON:-$FASTWAM_ENV/bin/python}
run_id=$(date -u +%Y%m%dT%H%M%S-%N)
output="$artifacts_root/inference/$run_id-libero-goal-7"
mkdir -p -m 700 "$artifacts_root/inference"
mkdir -m 700 "$output"

"$eval_python" experiments/libero/eval_libero_single.py \
    task=libero_uncond_2cam224_1e-4 \
    ckpt="$FASTWAM_ASSET_ROOT/checkpoints/libero_uncond_2cam224.pt" \
    gpu_id=0 \
    EVALUATION.task_suite_name=libero_goal \
    EVALUATION.task_id=7 \
    EVALUATION.num_trials=1 \
    EVALUATION.output_dir="$output" \
    EVALUATION.dataset_stats_path="$FASTWAM_ASSET_ROOT/checkpoints/libero_uncond_2cam224_dataset_stats.json" \
    2>&1 | tee "$output/worker.log"

"$FASTWAM_ENV/bin/python" "$TOOLS_DIR/verify_inference.py" \
    "$output" libero_goal 7 1 | tee "$output/verification.json"

latest_tmp="$artifacts_root/.latest-inference.$$"
printf '%s\n' "$output" >"$latest_tmp"
mv "$latest_tmp" "$artifacts_root/latest-inference.txt"
printf 'OK: inference smoke run verified at %s\n' "$output"
