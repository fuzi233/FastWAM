#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PYTHON=${PYTHON:-/export/code/sunxiaoquan/fastwam-baseline/env/bin/python}
VERIFY_INFERENCE="$ROOT/tools/verify_inference.py"
VERIFY_TRAINING_STATE="$ROOT/tools/verify_training_state.py"
CHECK_ASSETS="$ROOT/tools/check_assets.sh"

for validator in "$VERIFY_INFERENCE" "$VERIFY_TRAINING_STATE"; do
    if [[ ! -f "$validator" ]]; then
        printf 'FAIL: required validator is missing: %s\n' "$validator" >&2
        exit 1
    fi
done

TMP_DIR=$(mktemp -d)
trap 'rm -rf "$TMP_DIR"' EXIT
TESTS=0

expect_pass() {
    local name=$1
    shift
    TESTS=$((TESTS + 1))
    if ! "$@" >/dev/null 2>&1; then
        printf 'FAIL: expected pass: %s\n' "$name" >&2
        return 1
    fi
}

expect_fail() {
    local name=$1
    shift
    TESTS=$((TESTS + 1))
    if "$@" >/dev/null 2>&1; then
        printf 'FAIL: expected failure: %s\n' "$name" >&2
        return 1
    fi
}

check_asset_root_config() {
    # shellcheck source=../env.sh
    source "$ROOT/tools/env.sh"
    [[ "$FASTWAM_ASSET_ROOT" == /data/datasets/sunxiaoquan/FastWAM ]]
    [[ "$DIFFSYNTH_MODEL_BASE_PATH" == "$FASTWAM_ASSET_ROOT/models/Wan2.2-TI2V-5B" ]]
}

expect_pass "asset root uses the private FastWAM namespace" check_asset_root_config

write_result() {
    local output=$1 suite=$2 task_id=$3 trials=$4 successes=$5
    mkdir -p "$output/$suite"
    printf '{"task_suite":"%s","task_id":%s,"total_episodes":%s,"successes":%s}\n' \
        "$suite" "$task_id" "$trials" "$successes" \
        >"$output/$suite/gpu0_task${task_id}_results.json"
}

INFERENCE_DIR="$TMP_DIR/inference"
write_result "$INFERENCE_DIR" libero_spatial 2 5 3
expect_pass "valid inference result" \
    "$PYTHON" "$VERIFY_INFERENCE" "$INFERENCE_DIR" libero_spatial 2 5

write_result "$INFERENCE_DIR" wrong_suite 2 5 3
mv "$INFERENCE_DIR/wrong_suite/gpu0_task2_results.json" \
    "$INFERENCE_DIR/libero_spatial/gpu0_task2_results.json"
expect_fail "mismatched inference suite" \
    "$PYTHON" "$VERIFY_INFERENCE" "$INFERENCE_DIR" libero_spatial 2 5

write_result "$INFERENCE_DIR" libero_spatial 2 5 3
printf '%s\n' '{"task_suite":"libero_spatial","task_id":3,"total_episodes":5,"successes":3}' \
    >"$INFERENCE_DIR/libero_spatial/gpu0_task2_results.json"
expect_fail "mismatched inference task" \
    "$PYTHON" "$VERIFY_INFERENCE" "$INFERENCE_DIR" libero_spatial 2 5

write_result "$INFERENCE_DIR" libero_spatial 2 4 3
expect_fail "mismatched inference trials" \
    "$PYTHON" "$VERIFY_INFERENCE" "$INFERENCE_DIR" libero_spatial 2 5

write_result "$INFERENCE_DIR" libero_spatial 2 5 3
cp "$INFERENCE_DIR/libero_spatial/gpu0_task2_results.json" \
    "$INFERENCE_DIR/libero_spatial/gpu1_task2_results.json"
expect_fail "duplicate inference results" \
    "$PYTHON" "$VERIFY_INFERENCE" "$INFERENCE_DIR" libero_spatial 2 5
rm "$INFERENCE_DIR/libero_spatial/gpu1_task2_results.json"

rm "$INFERENCE_DIR/libero_spatial/gpu0_task2_results.json"
expect_fail "missing inference result" \
    "$PYTHON" "$VERIFY_INFERENCE" "$INFERENCE_DIR" libero_spatial 2 5

make_state() {
    local state_dir=$1 step=$2
    mkdir -p "$state_dir/pytorch_model"
    printf '{"global_step":%s,"epoch":0,"batch_in_epoch":0}\n' "$step" \
        >"$state_dir/trainer_state.json"
    printf 'scheduler\n' >"$state_dir/scheduler.bin"
    printf 'random state\n' >"$state_dir/random_states_0.pkl"
    printf 'model state\n' >"$state_dir/pytorch_model/mp_rank_00_model_states.pt"
    printf 'optimizer state\n' >"$state_dir/pytorch_model/mp_rank_00_optim_states.pt"
}

STATE_DIR="$TMP_DIR/state"
LOG="$TMP_DIR/train.log"
make_state "$STATE_DIR" 10
printf '%s\n' '[train] epoch=0 step=10/20 loss=0.1250 lr=1.00e-04' >"$LOG"
expect_pass "valid fresh training state" \
    "$PYTHON" "$VERIFY_TRAINING_STATE" "$STATE_DIR" 10 "$LOG" fresh

printf '%s\n' '{"global_step":9,"epoch":0,"batch_in_epoch":0}' \
    >"$STATE_DIR/trainer_state.json"
expect_fail "mismatched training global step" \
    "$PYTHON" "$VERIFY_TRAINING_STATE" "$STATE_DIR" 10 "$LOG" fresh

make_state "$STATE_DIR" 10
rm "$STATE_DIR/scheduler.bin"
expect_fail "missing required training state file" \
    "$PYTHON" "$VERIFY_TRAINING_STATE" "$STATE_DIR" 10 "$LOG" fresh

make_state "$STATE_DIR" 10
: >"$STATE_DIR/random_states_0.pkl"
expect_fail "empty required training state file" \
    "$PYTHON" "$VERIFY_TRAINING_STATE" "$STATE_DIR" 10 "$LOG" fresh

make_state "$STATE_DIR" 10
printf '%s\n' '[train] epoch=0 step=9/20 loss=0.1250' >"$LOG"
expect_fail "loss from a different step" \
    "$PYTHON" "$VERIFY_TRAINING_STATE" "$STATE_DIR" 10 "$LOG" fresh

printf '%s\n' '[train] epoch=0 step=10/20 lr=1.00e-04' >"$LOG"
expect_fail "missing loss at expected step" \
    "$PYTHON" "$VERIFY_TRAINING_STATE" "$STATE_DIR" 10 "$LOG" fresh

printf '%s\n' '[train]' 'epoch=0' 'step=10/20' 'loss=0.1250' >"$LOG"
expect_fail "split-line loss evidence" \
    "$PYTHON" "$VERIFY_TRAINING_STATE" "$STATE_DIR" 10 "$LOG" fresh

printf '%s\n' '[train] epoch=0 step=10/20 loss=NaN' >"$LOG"
expect_fail "NaN loss" \
    "$PYTHON" "$VERIFY_TRAINING_STATE" "$STATE_DIR" 10 "$LOG" fresh

printf '%s\n' '[train] epoch=0 step=10/20 loss=Inf' >"$LOG"
expect_fail "infinite loss" \
    "$PYTHON" "$VERIFY_TRAINING_STATE" "$STATE_DIR" 10 "$LOG" fresh

write_resumed_log() {
    local omit=${1:-none}
    : >"$LOG"
    [[ "$omit" == resume ]] || printf '%s\n' \
        'Resuming full training state from directory: /tmp/state' >>"$LOG"
    [[ "$omit" == dataloader ]] || printf '%s\n' \
        'Restored dataloader progress: epoch=0 batch_in_epoch=0 sample_offset=0' >>"$LOG"
    [[ "$omit" == accelerate ]] || printf '%s\n' \
        'Loaded accelerate training state from /tmp/state at step=10' >>"$LOG"
    printf '%s\n' '[train] epoch=0 step=10/20 loss=0.1250' >>"$LOG"
}

write_resumed_log
expect_pass "resumed state with all recovery evidence" \
    "$PYTHON" "$VERIFY_TRAINING_STATE" "$STATE_DIR" 10 "$LOG" resumed

for evidence in resume dataloader accelerate; do
    write_resumed_log "$evidence"
    expect_fail "resumed state missing $evidence evidence" \
        "$PYTHON" "$VERIFY_TRAINING_STATE" "$STATE_DIR" 10 "$LOG" resumed
done

FAKE_BIN="$TMP_DIR/fake-bin"
FAKE_PYTHON="$TMP_DIR/fake-python"
GPU_CALLS="$TMP_DIR/nvidia-smi.calls"
TORCH_CALLS="$TMP_DIR/torch-imported"
mkdir -p "$FAKE_BIN" "$FAKE_PYTHON"
cat >"$FAKE_BIN/nvidia-smi" <<'EOF'
#!/usr/bin/env bash
printf '%s\n' "$*" >>"${FAKE_NVIDIA_LOG:?}"
printf '%s' "${FAKE_NVIDIA_OUTPUT-}"
EOF
chmod +x "$FAKE_BIN/nvidia-smi"
cat >"$FAKE_PYTHON/torch.py" <<'EOF'
import os
from pathlib import Path

Path(os.environ["FAKE_TORCH_LOG"]).write_text("imported\n", encoding="utf-8")


class _Cuda:
    @staticmethod
    def device_count():
        return 1


cuda = _Cuda()
EOF

fake_gpu_env() {
    env \
        PATH="$FAKE_BIN:$PATH" \
        PYTHONPATH="$FAKE_PYTHON" \
        FAKE_NVIDIA_LOG="$GPU_CALLS" \
        FAKE_NVIDIA_OUTPUT="$1" \
        FAKE_TORCH_LOG="$TORCH_CALLS" \
        "${@:2}"
}

check_gpu_disabled() {
    rm -f "$GPU_CALLS" "$TORCH_CALLS"
    fake_gpu_env $'0, None\n' env CHECK_GPU=0 bash "$CHECK_ASSETS"
    [[ ! -e "$GPU_CALLS" && ! -e "$TORCH_CALLS" ]]
}

check_default_gpu_query() {
    local expected_args='-i 4 --query-gpu=memory.used,gpu_recovery_action --format=csv,noheader,nounits'
    rm -f "$GPU_CALLS" "$TORCH_CALLS"
    fake_gpu_env $'0, None\n' env -u CHECK_GPU bash "$CHECK_ASSETS"
    [[ "$(<"$GPU_CALLS")" == "$expected_args" && -e "$TORCH_CALLS" ]]
}

expect_safe_gpu_failure() {
    local name=$1 output=$2
    TESTS=$((TESTS + 1))
    rm -f "$GPU_CALLS" "$TORCH_CALLS"
    if fake_gpu_env "$output" env CHECK_GPU=1 bash "$CHECK_ASSETS" >/dev/null 2>&1; then
        printf 'FAIL: expected GPU preflight failure: %s\n' "$name" >&2
        return 1
    fi
    if [[ -e "$TORCH_CALLS" ]]; then
        printf 'FAIL: GPU preflight reached CUDA import: %s\n' "$name" >&2
        return 1
    fi
}

expect_pass "CHECK_GPU=0 skips nvidia-smi and CUDA import" check_gpu_disabled
expect_pass "default GPU query uses only the required arguments" check_default_gpu_query
expect_safe_gpu_failure "busy GPU" $'1025, None\n'
expect_safe_gpu_failure "GPU recovery action" $'0, Reset\n'
expect_safe_gpu_failure "missing GPU output comma" $'0 None\n'
expect_safe_gpu_failure "empty GPU output" ''
expect_safe_gpu_failure "non-numeric GPU memory" $'unknown, None\n'
expect_safe_gpu_failure "multiple GPU output lines" $'0, None\n0, None\n'

printf 'PASS: %d baseline tool checks\n' "$TESTS"
