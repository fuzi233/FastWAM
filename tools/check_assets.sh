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

require_dir /data/models/Wan2.2-TI2V-5B
require_file /data/models/Motus_FastWAM/pretrain_model/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt
require_file /data/models/fastwam/libero_uncond_2cam224.pt
require_file /data/models/fastwam/libero_uncond_2cam224_dataset_stats.json
require_dir /data/datasets/fastwam/text_embeds_cache/libero

for dataset in \
    libero_spatial_no_noops_lerobot \
    libero_object_no_noops_lerobot \
    libero_goal_no_noops_lerobot \
    libero_10_no_noops_lerobot; do
    require_dir "/data/datasets/fastwam/$dataset"
done

"$PYTHON" -m pip check

CHECK_GPU=${CHECK_GPU:-1}
[[ "$CHECK_GPU" == 0 || "$CHECK_GPU" == 1 ]] || fail "CHECK_GPU must be 0 or 1"
if [[ "$CHECK_GPU" == 1 ]]; then
    gpu_status=$(nvidia-smi -i 4 \
        --query-gpu=memory.used,gpu_recovery_action \
        --format=csv,noheader,nounits) || fail "nvidia-smi health query failed for GPU 4"

    [[ -n "$gpu_status" ]] || fail "nvidia-smi returned empty output for GPU 4"
    if [[ "$gpu_status" == *$'\n'* || "$gpu_status" == *$'\r'* ]]; then
        fail "nvidia-smi must return exactly one line for GPU 4"
    fi
    if [[ "$gpu_status" != *,* || "${gpu_status#*,}" == *,* ]]; then
        fail "nvidia-smi must return exactly two fields for GPU 4"
    fi

    IFS=, read -r memory_used recovery_action <<<"$gpu_status"
    memory_used=${memory_used#"${memory_used%%[![:space:]]*}"}
    memory_used=${memory_used%"${memory_used##*[![:space:]]}"}
    recovery_action=${recovery_action#"${recovery_action%%[![:space:]]*}"}
    recovery_action=${recovery_action%"${recovery_action##*[![:space:]]}"}

    [[ "$memory_used" =~ ^[0-9]+$ ]] || fail "invalid GPU 4 memory.used value: $memory_used"
    [[ "$recovery_action" == None ]] || fail "GPU 4 recovery action is not None: $recovery_action"
    if ((memory_used > 1024)) && [[ "${ALLOW_BUSY_GPU:-0}" != 1 ]]; then
        fail "GPU 4 is busy (${memory_used} MiB used; set ALLOW_BUSY_GPU=1 to override)"
    fi

    "$PYTHON" -c 'import torch; assert torch.cuda.device_count() == 1, f"expected exactly one visible CUDA device, got {torch.cuda.device_count()}"'
fi

printf 'OK: baseline assets and environment checks passed\n'
