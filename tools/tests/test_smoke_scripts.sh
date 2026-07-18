#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
INFER_SCRIPT="$ROOT/tools/smoke_infer.sh"
TRAIN_SCRIPT="$ROOT/tools/smoke_train_resume.sh"

for script in "$INFER_SCRIPT" "$TRAIN_SCRIPT"; do
    if [[ ! -x "$script" ]]; then
        printf 'FAIL: required smoke script is missing or not executable: %s\n' "$script" >&2
        exit 1
    fi
done

TMP_DIR=$(mktemp -d)
trap 'rm -rf "$TMP_DIR"' EXIT
FAKE_BIN="$TMP_DIR/fake-bin"
ARTIFACTS="$TMP_DIR/artifacts"
CALLS="$TMP_DIR/calls.log"
mkdir -p "$FAKE_BIN" "$ARTIFACTS"

cat >"$FAKE_BIN/eval-python" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
printf 'eval %s\n' "$*" >>"${FAKE_CALLS:?}"
output=
for arg in "$@"; do
    case "$arg" in
        EVALUATION.output_dir=*) output=${arg#*=} ;;
    esac
done
[[ -n "$output" ]]
mkdir -p "$output/libero_goal/videos"
printf '%s\n' '{"task_suite":"libero_goal","task_id":7,"total_episodes":1,"successes":1}' \
    >"$output/libero_goal/gpu0_task7_results.json"
EOF

cat >"$FAKE_BIN/accelerate" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
printf 'train %s\n' "$*" >>"${FAKE_CALLS:?}"
output=
max_steps=
resume=
for arg in "$@"; do
    case "$arg" in
        output_dir=*) output=${arg#*=} ;;
        max_steps=*) max_steps=${arg#*=} ;;
        resume=*) resume=${arg#*=} ;;
    esac
done
[[ -n "$output" && -n "$max_steps" ]]
step_tag=$(printf 'step_%06d' "$max_steps")
state="$output/checkpoints/state/$step_tag"
mkdir -p "$state/pytorch_model" "$output/checkpoints/weights"
printf '{"global_step":%s,"epoch":0,"batch_in_epoch":1}\n' "$max_steps" >"$state/trainer_state.json"
printf scheduler >"$state/scheduler.bin"
printf random >"$state/random_states_0.pkl"
printf model >"$state/pytorch_model/mp_rank_00_model_states.pt"
printf optimizer >"$state/pytorch_model/bf16_zero_pp_rank_0_mp_rank_00_optim_states.pt"
printf weights >"$output/checkpoints/weights/$step_tag.pt"
if [[ -n "$resume" ]]; then
    printf 'Resuming full training state from directory: %s\n' "$resume"
    printf 'Restored dataloader progress: epoch=0 batch_in_epoch=1 sample_offset=1\n'
fi
printf '[train] epoch=0 step=%s/%s loss=0.1250 lr=1.00e-04\n' "$max_steps" "$max_steps"
EOF
chmod +x "$FAKE_BIN/eval-python" "$FAKE_BIN/accelerate"

CHECK_GPU=0 \
FASTWAM_ARTIFACTS_OVERRIDE="$ARTIFACTS" \
FASTWAM_EVAL_PYTHON="$FAKE_BIN/eval-python" \
FAKE_CALLS="$CALLS" \
bash "$INFER_SCRIPT"

inference_output=$(<"$ARTIFACTS/latest-inference.txt")
[[ -s "$inference_output/verification.json" ]]
grep -Fq 'experiments/libero/eval_libero_single.py' "$CALLS"
grep -Fq 'ckpt=/data/datasets/sunxiaoquan/FastWAM/checkpoints/libero_uncond_2cam224.pt' "$CALLS"
grep -Fq 'EVALUATION.dataset_stats_path=/data/datasets/sunxiaoquan/FastWAM/checkpoints/libero_uncond_2cam224_dataset_stats.json' "$CALLS"

CHECK_GPU=0 \
FASTWAM_ARTIFACTS_OVERRIDE="$ARTIFACTS" \
FASTWAM_ACCELERATE="$FAKE_BIN/accelerate" \
FAKE_CALLS="$CALLS" \
bash "$TRAIN_SCRIPT"

training_output=$(<"$ARTIFACTS/latest-training.txt")
[[ -s "$training_output/step1/verification.json" ]]
[[ -s "$training_output/step2/verification.json" ]]
[[ $(grep -c '^train ' "$CALLS") -eq 2 ]]
grep -Fq 'save_every=0' "$CALLS"
grep -Fq '+data.train.pretrained_norm_stats=/data/datasets/sunxiaoquan/FastWAM/checkpoints/libero_uncond_2cam224_dataset_stats.json' "$CALLS"
grep -Fq "resume=$training_output/step1/checkpoints/state/step_000001" "$CALLS"

printf 'PASS: smoke script integration checks\n'
