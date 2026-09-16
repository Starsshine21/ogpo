# OGPO 快速上手指南

本文档面向第一次使用本仓库的用户，目标是在一台新的 GPU 机器上跑通：

1. critic 训练；
2. critic 评测；
3. OGPO actor 训练；
4. 固定协议策略评测；
5. Best-of-4 / QGF 推理。

## 1. 仓库内容

本仓库是独立开源版本，不依赖原始实验 monorepo。

```text
src/ogpo/                  核心 Python 包
scripts/                   训练、构数据、评测、推理脚本
configs/ogpo/              critic / actor 配置
protocols/                 RoboTwin 固定 manifest 与评测协议
third_party/openpi/        本项目使用的 OpenPI 源码快照
tests/                     单元测试
docs/                      算法与实验文档
```

以下内容没有提交，需要你在本机准备：

- PI0.5、Gemma、SigLIP checkpoint；
- RoboTwin 原始 episode；
- 预处理后的 replay `.pt`；
- 训练 checkpoint、日志、tensorboard 输出。

## 2. 安装

```bash
git clone https://github.com/Starsshine21/ogpo.git
cd ogpo
conda env create -f environment.yml
conda activate ogpo
python scripts/check_install.py
```

如果已经有环境，也可以手动安装：

```bash
pip install -e .
pip install -e ./third_party/openpi
pip install -e ./third_party/openpi/packages/openpi-client
pip install pytest
python scripts/check_install.py
```

安装检查通过时会输出：

```text
OGPO_INSTALL_CHECK_PASS
```

## 3. 准备模型文件

默认配置假设以下目录存在：

```text
checkpoints/models/gemma-3-270m/
checkpoints/models/siglip2-so400m-patch14-224-fixed/
checkpoints/pi05/model_clean50/
```

你也可以把模型放到其他位置，然后修改 YAML：

- `critic.backbone.gemma_path`
- `critic.backbone.siglip_path`
- `flow.checkpoint_dir`
- `training.critic_checkpoint`

这些大文件不要提交到 Git。

## 4. 准备 replay 数据

OGPO 训练使用 action-chunk transition。每条样本至少需要：

- 当前 observation / proprioception；
- `[horizon, action_dim]` action chunk；
- execution mask / executed length；
- chunk return、discount、done、success；
- episode ID、timestep、task ID、language；
- 图像和下一状态图像（多模态 critic 需要）。

从 RoboTwin dense episodes 构建 replay：

```bash
python scripts/build_robotwin_multitask_streaming_replay.py \
  --input-root /path/to/dense_rollouts \
  --config configs/ogpo/robotwin_multitask10_divl_vmean_headonly_taskoutcomebalanced_20k.yaml \
  --output-prefix outputs/ogpo/replays/my_mixed_replay \
  --expected-episodes 1000 \
  --num-train-shards 4
```

如果要复现我们的 mixed-1000 bootstrap 数据划分，先按照
`protocols/robotwin_mixed1000/data_transfer_manifest.json` 准备原始 episode，
再运行：

```bash
python scripts/prepare_mixed1000_bootstrap.py --build
```

原始 episode 很大，没有放入 GitHub。协议目录里包含 manifest、split、
bootstrap mask 和 noise 规则，不包含原始图像/动作数据。

## 5. 训练 critic

Categorical Q（推荐实验路线）：

```bash
python scripts/train_udivl_critic.py \
  --config configs/ogpo/robotwin_mixed1000_bootstrap_catq_shared32_20k.yaml
```

Scalar Q 对照：

```bash
python scripts/train_udivl_critic.py \
  --config configs/ogpo/robotwin_mixed1000_bootstrap_scalar_shared32_20k.yaml
```

四进程训练：

```bash
torchrun --nproc_per_node=4 scripts/train_udivl_critic.py \
  --config configs/ogpo/robotwin_mixed1000_bootstrap_catq_shared32_20k.yaml
```

只跑 2000 步：

```bash
python scripts/train_udivl_critic.py \
  --config configs/ogpo/robotwin_mixed1000_bootstrap_catq_shared32_20k.yaml \
  --critic-steps 2000
```

恢复训练：

```bash
python scripts/train_udivl_critic.py \
  --config configs/ogpo/robotwin_mixed1000_bootstrap_catq_shared32_20k.yaml \
  --resume outputs/ogpo/checkpoints/.../latest.pt
```

训练输出默认写入：

```text
outputs/ogpo/checkpoints/
outputs/ogpo/*_metrics.jsonl
outputs/ogpo/tensorboard/
```

## 6. 评测 critic

### 6.1 OGPO critic evaluator v2

需要一个固定的 base-actor candidate cache。多个 critic 使用同一个 cache，
保证公平比较：

```bash
python scripts/evaluate_ogpo_critic_v2.py \
  --candidate-cache /path/to/candidate_cache.pt \
  --output-dir outputs/ogpo/evaluations/my_eval \
  --critic-spec CatQ::/path/to/catq.pt::configs/ogpo/robotwin_mixed1000_bootstrap_catq_shared32_20k.yaml \
  --critic-spec Scalar::/path/to/scalar.pt::configs/ogpo/robotwin_mixed1000_bootstrap_scalar_shared32_20k.yaml
```

主要输出：

- value fidelity：MC Spearman、RMSE、terminal error；
- candidate reliability：Top-1 agreement、pairwise consistency、margin；
- advantage density：CA nonzero、median |A|；
- ensemble diagnostics：Q-head correlation、ensemble std。

### 6.2 CARL v2

先冻结 candidate bank：

```bash
python scripts/prepare_carl.py \
  --candidate-cache /path/to/candidate_cache.pt \
  --output-dir outputs/ogpo/carl/bank_v1 \
  --states-per-task 10 \
  --terminal-replay /path/to/terminal_replay.pt \
  --logged-replay /path/to/logged_replay.pt
```

再评测：

```bash
python scripts/evaluate_carl.py \
  --root outputs/ogpo/carl/bank_v1 \
  --report-dir outputs/ogpo/carl/report_v1 \
  --critic-spec CatQ::/path/to/catq.pt::configs/ogpo/robotwin_mixed1000_bootstrap_catq_shared32_20k.yaml \
  --critic-spec Scalar::/path/to/scalar.pt::configs/ogpo/robotwin_mixed1000_bootstrap_scalar_shared32_20k.yaml
```

CARL v2 使用 Sparse Reward Policy Alignment，不会在仿真器里 rollout
candidate，因此速度比真实环境 correctness 评测快很多。

## 7. 训练 OGPO actor

先确认 actor config 中的 replay 路径和 PI0.5 base checkpoint 正确，然后：

```bash
python scripts/train_full_ogpo.py \
  --config configs/ogpo/robotwin_mixed1000_catq9k_actor_2k.yaml \
  --critic-checkpoint /path/to/catq_critic.pt
```

短程 smoke：

```bash
python scripts/train_full_ogpo.py \
  --config configs/ogpo/robotwin_mixed1000_catq9k_actor_2k.yaml \
  --critic-checkpoint /path/to/catq_critic.pt \
  --actor-steps 10 --smoke
```

恢复：

```bash
python scripts/train_full_ogpo.py \
  --config configs/ogpo/robotwin_mixed1000_catq9k_actor_2k.yaml \
  --resume outputs/ogpo/checkpoints/.../latest.pt
```

当前 mixed-10 actor recipe：

- critic frozen；
- `G=4`；
- task-balanced cycle；
- conservative advantage；
- 定期保存 actor checkpoint。

## 8. 固定协议评测 actor

评测协议在：

```text
protocols/robotwin_mixed1000/evaluation/
```

先检查协议包：

```bash
cd protocols/robotwin_mixed1000/evaluation
python scripts/bundle_tools.py validate --bundle-root "$PWD"
```

设置路径：

```bash
export OGPO_ROOT=/path/to/ogpo
export PI05_ROOT=/path/to/pi05
export ROBOTWIN_ROOT=/path/to/RoboTwin
export PI05_CHECKPOINT_DIR=/path/to/model_clean50
export PYTHON_BIN=/path/to/python
```

评测 base：

```bash
export OUTPUT_ROOT=/path/to/results/base
export POLICY_LABEL=base
unset ACTOR_CHECKPOINT
bash scripts/run_policy_strict100.sh
```

评测 OGPO actor：

```bash
export OUTPUT_ROOT=/path/to/results/actor
export POLICY_LABEL=ogpo_actor
export ACTOR_CHECKPOINT=/path/to/actor.pt
bash scripts/run_policy_strict100.sh
```

所有模型必须使用同一 manifest、环境 seed、prompt 和 policy noise 协议。

## 9. Best-of-4 与 QGF

这两个方法都不训练模型，只在推理时使用 critic。

Best-of-4：生成 4 个完整 action chunks，用 10 个 Q head 的均值选最高：

```bash
python scripts/collect_robotwin_dense_rollouts.py \
  --task-name click_bell \
  --task-config demo_clean \
  --train-config-name pi05_robotwin2_clean50_full \
  --model-name model_clean50 \
  --checkpoint-id 20000 \
  --pi05-checkpoint-dir "$PI05_CHECKPOINT_DIR" \
  --episode-manifest /path/to/click_bell.json \
  --trust-episode-manifest \
  --seed 0 --num-episodes 13 \
  --output-dir outputs/robotwin_eval/bestof4/click_bell/shard_00 \
  --bestofn-critic-checkpoint /path/to/catq_critic.pt \
  --bestofn-critic-config configs/ogpo/robotwin_mixed1000_bootstrap_catq_shared32_20k.yaml \
  --bestofn-group-size 4
```

QGF：单候选，在 flow 去噪过程中用 Q 对 clean-action estimate 的梯度引导：

```bash
python scripts/collect_robotwin_dense_rollouts.py \
  --task-name click_bell \
  --task-config demo_clean \
  --train-config-name pi05_robotwin2_clean50_full \
  --model-name model_clean50 \
  --checkpoint-id 20000 \
  --pi05-checkpoint-dir "$PI05_CHECKPOINT_DIR" \
  --episode-manifest /path/to/click_bell.json \
  --trust-episode-manifest \
  --seed 0 --num-episodes 13 \
  --output-dir outputs/robotwin_eval/qgf/click_bell/shard_00 \
  --qgf-critic-checkpoint /path/to/catq_critic.pt \
  --qgf-critic-config configs/ogpo/robotwin_mixed1000_bootstrap_catq_shared32_20k.yaml \
  --qgf-strength 1.0
```

## 10. 常见问题

### 找不到 Gemma / SigLIP

检查 YAML 中：

```yaml
critic:
  backbone:
    gemma_path: ...
    siglip_path: ...
```

### 找不到 replay

检查 YAML 中：

```yaml
data:
  dataset_path: ...
  dataset_paths: ...
  validation_path: ...
```

路径都相对仓库根目录解析，也可以用绝对路径。

### actor 训练找不到 critic

确认 `--critic-checkpoint` 或 YAML 里的
`training.critic_checkpoint` 指向训练好的 `.pt`。

### 固定评测路径错误

确认三个环境变量：

```bash
export OGPO_ROOT=/path/to/ogpo
export PI05_ROOT=/path/to/pi05
export ROBOTWIN_ROOT=/path/to/RoboTwin
```

### 想确认没有提交大文件

```bash
git status --short
git ls-files | grep -E '\.(pt|pth|safetensors|npz)$' || true
```

后一条命令不应该输出训练 checkpoint 或原始 `frames.npz`。

