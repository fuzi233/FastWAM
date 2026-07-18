#!/usr/bin/env bash
set -euo pipefail

TOOLS_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=env.sh
source "$TOOLS_DIR/env.sh"

fail() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

require_file() {
    [[ -f "$1" ]] || fail "required file is missing: $1"
    [[ -s "$1" ]] || fail "required file is empty: $1"
}

require_dir() {
    [[ -d "$1" ]] || fail "required directory is missing: $1"
}

PYTHON="$FASTWAM_ENV/bin/python"
[[ -x "$PYTHON" ]] || fail "target environment Python is missing or not executable: $PYTHON"

[[ "$FASTWAM_GPU_LIST" =~ ^[0-4](,[0-4])*$ ]] \
    || fail "FASTWAM_GPU_LIST must be a comma-separated selection from physical GPUs 0-4"
IFS=, read -r -a gpu_ids <<<"$FASTWAM_GPU_LIST"
declare -A selected_gpu_ids=()
for gpu_id in "${gpu_ids[@]}"; do
    [[ -z "${selected_gpu_ids[$gpu_id]:-}" ]] \
        || fail "duplicate physical GPU in FASTWAM_GPU_LIST: $gpu_id"
    selected_gpu_ids[$gpu_id]=1
done

require_dir "$DIFFSYNTH_MODEL_BASE_PATH"
require_file "$FASTWAM_ASSET_ROOT/checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt"
require_file "$FASTWAM_ASSET_ROOT/checkpoints/libero_uncond_2cam224.pt"
require_file "$FASTWAM_ASSET_ROOT/checkpoints/libero_uncond_2cam224_dataset_stats.json"
require_dir "$FASTWAM_ASSET_ROOT/text_embeds_cache/libero"

for dataset in \
    libero_spatial_no_noops_lerobot \
    libero_object_no_noops_lerobot \
    libero_goal_no_noops_lerobot \
    libero_10_no_noops_lerobot; do
    require_dir "$FASTWAM_ASSET_ROOT/datasets/$dataset"
done

"$PYTHON" -m pip check

CHECK_GPU=${CHECK_GPU:-1}
[[ "$CHECK_GPU" == 0 || "$CHECK_GPU" == 1 ]] || fail "CHECK_GPU must be 0 or 1"
if [[ "$CHECK_GPU" == 1 ]]; then
    for gpu_id in "${gpu_ids[@]}"; do
        gpu_status=$(nvidia-smi -i "$gpu_id" \
            --query-gpu=memory.used,gpu_recovery_action \
            --format=csv,noheader,nounits) \
            || fail "nvidia-smi health query failed for physical GPU $gpu_id"

        [[ -n "$gpu_status" ]] \
            || fail "nvidia-smi returned empty output for physical GPU $gpu_id"
        if [[ "$gpu_status" == *$'\n'* || "$gpu_status" == *$'\r'* ]]; then
            fail "nvidia-smi must return exactly one line for physical GPU $gpu_id"
        fi
        if [[ "$gpu_status" != *,* || "${gpu_status#*,}" == *,* ]]; then
            fail "nvidia-smi must return exactly two fields for physical GPU $gpu_id"
        fi

        IFS=, read -r memory_used recovery_action <<<"$gpu_status"
        memory_used=${memory_used#"${memory_used%%[![:space:]]*}"}
        memory_used=${memory_used%"${memory_used##*[![:space:]]}"}
        recovery_action=${recovery_action#"${recovery_action%%[![:space:]]*}"}
        recovery_action=${recovery_action%"${recovery_action##*[![:space:]]}"}

        [[ "$memory_used" =~ ^[0-9]+$ ]] \
            || fail "invalid physical GPU $gpu_id memory.used value: $memory_used"
        [[ "$recovery_action" == None ]] \
            || fail "physical GPU $gpu_id recovery action is not None: $recovery_action"
        if ((memory_used > 1024)) && [[ "${ALLOW_BUSY_GPU:-0}" != 1 ]]; then
            fail "physical GPU $gpu_id is busy (${memory_used} MiB used; set ALLOW_BUSY_GPU=1 to override)"
        fi
    done

    FASTWAM_EXPECTED_GPU_COUNT=${#gpu_ids[@]} "$PYTHON" -c \
        'import os, torch; expected = int(os.environ["FASTWAM_EXPECTED_GPU_COUNT"]); actual = torch.cuda.device_count(); assert actual == expected, f"expected {expected} visible CUDA devices, got {actual}"'
fi

printf 'OK: baseline assets and environment checks passed\n'
