#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PYTHON=${PYTHON:-/export/code/sunxiaoquan/fastwam-baseline/env/bin/python}
VERIFY_INFERENCE="$ROOT/tools/verify_inference.py"
VERIFY_TRAINING_STATE="$ROOT/tools/verify_training_state.py"

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

printf 'PASS: %d baseline tool checks\n' "$TESTS"
