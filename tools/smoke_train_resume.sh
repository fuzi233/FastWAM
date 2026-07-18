#!/usr/bin/env bash
set -euo pipefail

TOOLS_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=env.sh
source "$TOOLS_DIR/env.sh"

cd "$FASTWAM_REPO"
bash "$TOOLS_DIR/check_assets.sh"

artifacts_root=${FASTWAM_ARTIFACTS_OVERRIDE:-$FASTWAM_ARTIFACTS}
accelerate_bin=${FASTWAM_ACCELERATE:-$FASTWAM_ENV/bin/accelerate}
run_id=$(date -u +%Y%m%dT%H%M%S-%N)
run_root="$artifacts_root/training/$run_id-step1-resume-step2"
step1_output="$run_root/step1"
step2_output="$run_root/step2"
mkdir -p -m 700 "$artifacts_root/training"
mkdir -p -m 700 "$step1_output" "$step2_output"

common_overrides=(
    task=libero_uncond_2cam224_1e-4
    batch_size=1
    num_workers=0
    log_every=1
    save_every=0
    eval_every=0
    wandb.enabled=false
    model.mot_checkpoint_mixed_attn=true
    "+data.train.pretrained_norm_stats=$FASTWAM_ASSET_ROOT/checkpoints/libero_uncond_2cam224_dataset_stats.json"
)

"$accelerate_bin" launch \
    --config_file scripts/accelerate_configs/accelerate_zero1_ds.yaml \
    --num_processes 1 \
    scripts/train.py \
    "${common_overrides[@]}" \
    output_dir="$step1_output" \
    max_steps=1 \
    2>&1 | tee "$step1_output/train.log"

step1_state="$step1_output/checkpoints/state/step_000001"
step1_weights="$step1_output/checkpoints/weights/step_000001.pt"
[[ -s "$step1_weights" ]] || { printf 'ERROR: missing step 1 weights: %s\n' "$step1_weights" >&2; exit 1; }
"$FASTWAM_ENV/bin/python" "$TOOLS_DIR/verify_training_state.py" \
    "$step1_state" 1 "$step1_output/train.log" fresh \
    | tee "$step1_output/verification.json"

"$accelerate_bin" launch \
    --config_file scripts/accelerate_configs/accelerate_zero1_ds.yaml \
    --num_processes 1 \
    scripts/train.py \
    "${common_overrides[@]}" \
    output_dir="$step2_output" \
    max_steps=2 \
    resume="$step1_state" \
    2>&1 | tee "$step2_output/train.log"

step2_state="$step2_output/checkpoints/state/step_000002"
step2_weights="$step2_output/checkpoints/weights/step_000002.pt"
[[ -s "$step2_weights" ]] || { printf 'ERROR: missing step 2 weights: %s\n' "$step2_weights" >&2; exit 1; }
"$FASTWAM_ENV/bin/python" "$TOOLS_DIR/verify_training_state.py" \
    "$step2_state" 2 "$step2_output/train.log" resumed \
    | tee "$step2_output/verification.json"

latest_tmp="$artifacts_root/.latest-training.$$"
printf '%s\n' "$run_root" >"$latest_tmp"
mv "$latest_tmp" "$artifacts_root/latest-training.txt"
printf 'OK: training resume smoke run verified at %s\n' "$run_root"
