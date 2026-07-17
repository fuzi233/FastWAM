#!/usr/bin/env bash
set -euo pipefail

export FASTWAM_BASELINE_ROOT=/export/code/sunxiaoquan/fastwam-baseline
export FASTWAM_REPO="$FASTWAM_BASELINE_ROOT/repo"
export FASTWAM_ENV="$FASTWAM_BASELINE_ROOT/env"
export FASTWAM_ARTIFACTS="$FASTWAM_BASELINE_ROOT/artifacts"
export CUDA_VISIBLE_DEVICES=4
export DIFFSYNTH_MODEL_BASE_PATH=/data/models/Wan2.2-TI2V-5B
export DIFFSYNTH_SKIP_DOWNLOAD=true
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1
