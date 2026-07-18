# FastWAM 隔离基线

本分支用于在 `ELU_H100` 上复现和扩展 FastWAM，同时与已有代码、环境、模型、数据和运行产物隔离。

## 固定版本与目录

- 官方基线提交：`45d8e1458921d83f8ad6cf9ce993d371208dabd0`
- 维护分支：`codex/fastwam-baseline`
- 代码：`/export/code/sunxiaoquan/fastwam-baseline/repo`
- Conda 前缀：`/export/code/sunxiaoquan/fastwam-baseline/env`
- 资产根：`/data/datasets/sunxiaoquan/FastWAM`
- 运行产物：`/export/code/sunxiaoquan/fastwam-baseline/artifacts`
- 安装与迁移证据：`/export/code/sunxiaoquan/fastwam-baseline/setup-logs`

进程固定设置 `CUDA_VISIBLE_DEVICES=4`，因此物理 GPU 4 在 FastWAM 进程内显示为逻辑 GPU 0。不得改用 GPU 0–3 或 GPU 5–7，也不得通过 `ALLOW_BUSY_GPU=1` 抢占已使用的 GPU 4。

## 资产布局

```text
/data/datasets/sunxiaoquan/FastWAM/
├── models/Wan2.2-TI2V-5B/
├── checkpoints/
│   ├── ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt
│   ├── libero_uncond_2cam224.pt
│   └── libero_uncond_2cam224_dataset_stats.json
├── datasets/
│   ├── libero_spatial_no_noops_lerobot/
│   ├── libero_object_no_noops_lerobot/
│   ├── libero_goal_no_noops_lerobot/
│   └── libero_10_no_noops_lerobot/
└── text_embeds_cache/libero/
```

这些资产由原路径复制后经过路径/类型清单、逐文件 checksum 和大文件 SHA-256 校验。旧源当前保留，不得从本基线删除或修改。repo 内使用绝对符号链接；“只读复用”是本基线的行为约束，不是操作系统级只读挂载，因此每次重要运行后仍需核对共享资产未被修改。

## 运行前检查

```bash
cd /export/code/sunxiaoquan/fastwam-baseline/repo
bash tools/tests/test_baseline_tools.sh
bash tools/tests/test_smoke_scripts.sh
bash tools/check_assets.sh
```

`tools/check_assets.sh` 会检查全部资产、独立环境的依赖一致性以及物理 GPU 4。GPU recovery action 非 `None`、输出格式异常或显存占用超过 1024MiB时都会安全失败。

仅检查环境和资产、不查询或导入 GPU：

```bash
CHECK_GPU=0 bash tools/check_assets.sh
```

## 单任务 LIBERO 推理

```bash
bash tools/smoke_infer.sh
```

入口直接调用官方 `experiments/libero/eval_libero_single.py`，任务固定为 `libero_goal` task 7，运行一次 trial。最新通过机器校验的产物路径记录在：

```text
/export/code/sunxiaoquan/fastwam-baseline/artifacts/latest-inference.txt
```

基础设施通过条件是官方 worker 正常退出，且唯一结果 JSON 的 suite、task、trial 和 success 值域正确。策略实际成功或失败必须按 JSON 如实单独报告。

## 训练 step 1 与恢复到 step 2

```bash
bash tools/smoke_train_resume.sh
```

脚本按顺序启动两个独立 Accelerate 进程：

1. 第一个进程执行真实 forward、backward 和 optimizer step 1，保存权重及完整 Accelerate/DeepSpeed 状态。
2. 第一个进程退出后，第二个进程从 step 1 state 目录恢复 optimizer、scheduler、随机状态和 dataloader 进度，再执行 step 2。

最新通过机器校验的运行根目录记录在：

```text
/export/code/sunxiaoquan/fastwam-baseline/artifacts/latest-training.txt
```

状态验证要求有限 loss、正确 `global_step`、非空权重，以及非空 model、optimizer、scheduler、random 和 trainer state。完整恢复日志必须包含官方可达分支的 `Resuming full training state from directory` 与 `Restored dataloader progress`。

## 禁止事项

- 不修改 `/export/ra/sunxiaoquan/FastWAM` 或其现有工作区。
- 不向 `/export/ra/sunxiaoquan` 或本机写入模型、checkpoint 或数据集。
- 不删除旧资产，不改共享资产权限，不停止其他用户的 GPU 进程。
- 不修改 shell 启动文件来激活环境；始终使用独立前缀的绝对解释器路径。
- 不把实验提交推到官方 `origin`；代码推送到 `fork` remote。

## 依赖或代码升级

任何升级都必须在隔离分支和隔离 Conda 前缀中完成，并依次通过：

```bash
/export/code/sunxiaoquan/fastwam-baseline/env/bin/python -m pip check
bash tools/tests/test_baseline_tools.sh
bash tools/tests/test_smoke_scripts.sh
bash tools/smoke_infer.sh
bash tools/smoke_train_resume.sh
```

不得用“可以导入”代替推理与完整训练恢复验证。
