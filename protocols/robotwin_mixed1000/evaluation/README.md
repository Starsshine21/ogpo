# 当前混合10任务：固定协议策略成功率评测

每个模型10任务×100episodes。所有模型使用同一环境seed、完整prompt、每episode policy noise seed，以及每chunk相同的初始噪声生成方法。
宏平均逐任务等权。不修改或重抽manifest，不按模型替换失败seed。

## 任务与历史 Base reference

| Task | Base成功率 | 固定协议来源 | 当前训练split环境seed重叠数 |
| --- | ---: | --- | ---: |
| pick_dual_bottles | 61% | 旧正式Base评测 | 72 |
| handover_mic | 66% | 旧正式Base评测 | 74 |
| click_bell | 52% | 旧正式Base评测 | 82 |
| dump_bin_bigbin | 70% | 旧正式Base评测 | 76 |
| open_laptop | 45% | 旧正式Base评测 | 82 |
| click_alarmclock | 65% | 旧正式Base评测 | 85 |
| place_a2b_right | 39% | 旧正式Base评测 | 84 |
| adjust_bottle | 70% | 固定noise的Base采集 | 85 |
| place_object_stand | 77% | 固定noise的Base采集 | 83 |
| press_stapler | 62% | 固定noise的Base采集 | 77 |
| **Macro** | **60.7%** | 当前混合10任务 | **800** |

**全部1000个环境seed均与当前1000条采集数据重叠，其中800个在训练split。**
这套协议衡量相同训练场景上的策略改进，不能作为未见场景泛化证据。旧7任务换成固定noise评测，不会使其环境seed自动变成unseen。新增3任务的训练与评测还复用了相同noise。
旧表53.1%属于另一组10任务，不得与这里60.7%直接算训练提升。

## 文件结构

- `protocol.json`：任务、步数上限、policy设置和noise协议。
- `manifests/<task>.json`：**真正用于本轮成功率评测**的10份manifest；全部1000条noise seed非空。
- `base_reference/<task>/shard_*/raw_rollouts/episode_*/meta.json`：已核对manifest一致的1000份历史Base结果，可用于逐episode配对；不含images/actions。
- `BASE_REFERENCE.json`：逐任务Base成功率、macro、训练seed重叠审计。
- `PROVENANCE.json`：每份manifest的源路径及SHA256。7任务沿用旧policy evaluator原manifest，3任务沿用固定noise采集原manifest，均逐字节复制。
- `evaluator/`：便携collector、审计、实时进度脚本。
- `scripts/`：运行、验证、汇总、配对比较。

## 一致性要求

Base权重、归一化assets、RoboTwin及OpenPI代码、相机/机器人配置应一致。
Base使用 `model_clean50`；`train_config_name=pi05_robotwin2_clean50_full`；`demo_clean`；50-step action chunk；10 denoising steps。
noise维度 `[50,32]`，不是机器人执行action的14维。

```python
noise = np.random.default_rng(
    np.random.SeedSequence([policy_noise_seed, policy_chunk_id])
).standard_normal((50, 32), dtype=np.float32)
```

每episode的chunk计数从0开始。共享的是每个chunk index的初始noise，不保证不同策略产生相同状态、动作或episode长度。
评测默认metadata-only，输出足够审计成功率的记录；如要训练用dense数据，应使用父目录采集协议。
跨硬件/仿真器数值差异可能影响结果，建议另一台机器上同时重测Base和待比较Actor；历史60.7%仅是参考，不保证跨机器精确复现。

## 验证与执行

```bash
cd /path/to/evo-RL/robotwin_mixed1000_collection_protocol_v1/evaluation
python scripts/bundle_tools.py validate --bundle-root "$PWD"

export EVO_RL_ROOT=/path/to/evo-RL
export PI05_ROOT=/path/to/pi05
export PYTHON_BIN=/path/to/your/python
export PI05_CHECKPOINT_DIR=/path/to/pi05/model_clean50
export CUDA_VISIBLE_DEVICES=0,1,2,3
export OUTPUT_ROOT=/path/to/new_results/base
export POLICY_LABEL=base
unset ACTOR_CHECKPOINT
bash scripts/run_policy_strict100.sh
```

评测Actor：设置 `ACTOR_CHECKPOINT=/path/to/actor.pt`、新的 `OUTPUT_ROOT` 和 `POLICY_LABEL`，运行同一脚本。已由Slurm分配GPU时，不要手工覆盖CUDA_VISIBLE_DEVICES；脚本支持1–8张可见卡。
示例Slurm为源集群模板，目标集群必须修改账号、分区和资源请求。续跑设置 `RESUME_EXISTING=1`；不要两个任务共用同一输出目录。

```bash
python scripts/bundle_tools.py compare \
  --bundle-root "$PWD" \
  --base-root /path/to/new_results/base \
  --actor-root /path/to/new_results/actor \
  --actor-checkpoint /path/to/actor.pt \
  --output /path/to/new_results/comparison.json
```

同环境复核历史Base时也可将base-root设为 `$PWD/base_reference`。该对比工具读取逐episode metadata，不需要base images。
完整性校验在父目录执行 `sha256sum -c SHA256SUMS`。本次仅整理协议，没有提交新评测或训练。
