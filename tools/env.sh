#!/usr/bin/env bash
set -euo pipefail

export FASTWAM_BASELINE_ROOT=/export/code/sunxiaoquan/fastwam-baseline
export FASTWAM_REPO="$FASTWAM_BASELINE_ROOT/repo"
export FASTWAM_ENV="$FASTWAM_BASELINE_ROOT/env"
export FASTWAM_ARTIFACTS="$FASTWAM_BASELINE_ROOT/artifacts"
export FASTWAM_ASSET_ROOT=/data/datasets/sunxiaoquan/FastWAM
export FASTWAM_RUNTIME_ROOT="$FASTWAM_ASSET_ROOT/runtime_cache"
export HOME="$FASTWAM_RUNTIME_ROOT/home"
export HF_HOME="$FASTWAM_RUNTIME_ROOT/huggingface"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_CACHE="$HF_HOME/hub"
export XDG_CACHE_HOME="$FASTWAM_RUNTIME_ROOT/xdg"
export LIBERO_CONFIG_PATH="$FASTWAM_REPO/config/libero"
export FASTWAM_GPU_LIST=${FASTWAM_GPU_LIST:-4}
export CUDA_VISIBLE_DEVICES="$FASTWAM_GPU_LIST"
export DIFFSYNTH_MODEL_BASE_PATH="$FASTWAM_ASSET_ROOT/models/Wan2.2-TI2V-5B"
export DIFFSYNTH_SKIP_DOWNLOAD=true
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1

umask 077
