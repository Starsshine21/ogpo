# 当前混合10任务：真实采集 manifest / noise 协议

## 当前10任务的成功率评测入口

**要评测 Base / Actor，请进入 [`evaluation/`](evaluation/README.md)。**
该子目录提供全部10任务固定seed、prompt、noise的正式policy评测manifest、运行/审计/配对比较脚本，以及历史Base逐episode结果。
根目录的 `manifests/` 是实际采集记录（旧700条noise为空），**不可误用作固定噪声评测manifest**。
两套文件刻意分开，不覆盖采集来源。评测全部1000个环境seed与当前数据有重叠，其中800个是训练split；这是训练场景同协议比较，不是clean-unseen heldout。

这是 Bootstrap Scalar / CatQ 当前实际使用的1000条数据，不是旧的 `robotwin_10task_strict100_eval_v1` 评测集。完整1000条中800条参与训练，其余100 validation、100 heldout；split 不应重抽。

## 必须先知道的限制

- **新增3任务** adjust_bottle、place_object_stand、press_stapler：全部300条保存了 episode noise seed，可恢复每个 policy chunk 的初始噪声。
- **旧7任务** pick_dual_bottles、handover_mic、click_bell、dump_bin_bigbin、open_laptop、click_alarmclock、place_a2b_right：700条 `policy_noise_seed=null`。原 collector 调用 `policy.infer(obs)`，未显式传 noise；不能拿环境seed当noise seed，也不能把后来生成的噪声冒充原始噪声。
- 若要求**完全相同的训练数据**，应直接迁移原始 `frames.npz` / `meta.json`。仅同环境seed、prompt不能保证重采出同一条轨迹。即使固定noise，跨GPU、驱动、仿真器版本的轨迹也未必逐位一致。

## 文件

- `manifests/<task>.json`：从当前1000条实际 `meta.json` 导出的逐episode seed、完整prompt、noise seed、split、全局ID、源路径及已观察到的结果。
- `original_manifests/`：新增3任务采集时使用的原始manifest，未改内容。
- `noise/<task>/episode_<global_id>.npy`：新增300条已执行policy chunks的初始noise，shape `[num_policy_chunks,50,32]`，float32。**这是按保存的seed重建的数组，不是采集时额外录下的数组。**
- `split_manifest.json`、`bootstrap_masks.json`：当前数据划分和5member固定训练mask。全局episode_id必须保留。
- `data_transfer_manifest.json`：2000个原始数据文件的精确来源、目标相对路径和大小；本包不含原始轨迹。
- `rsync_source_files.txt`：迁移原始数据时使用的路径列表，相对于源机器的evo-RL根目录。
- `collector_snapshot/`：当前dense collector、原采集Slurm和replay builder快照。是打包时的代码快照，不能充当旧7任务的完整历史环境存档；包含旧机器路径，迁移前须修改。
- `protocol.json`：模型/噪声/数据格式。
- `SHA256SUMS`：协议包文件校验。

## 采集配置

Base PI0.5，`model_clean50`；无OGPO actor checkpoint。
`task_config=demo_clean`，`train_config_name=pi05_robotwin2_clean50_full`，`checkpoint_id=20000`。
每个policy chunk执行最多50步；flow的action维度为32，执行到机器人的action维度为14。不要生成 `[50,14]` 的flow noise。
采集为每环境步记录的dense overlapping actual-executed action windows，**不要开启metadata-only**。
原始数据gamma=.999；训练replay再按原builder转换到gamma=.995、n_step=2，不要直接把源discount解释成.995。

## 新增3任务的noise

```python
import numpy as np
noise = np.random.default_rng(
    np.random.SeedSequence([policy_noise_seed, policy_chunk_id])
).standard_normal((50, 32), dtype=np.float32)
```

PCG64；`policy_chunk_id` 每个episode从0开始，每次调用policy后递增。不是环境timestep；也不是每个dense transition生成一次。
必须在 `standard_normal` 内指定float32，不能先生成float64再cast。
将noise显式传给 `policy.infer(observation, noise=noise)`。环境seed、prompt、shard/episode_index全部按manifest，不重新抽取。
原始新增3任务按8shards分为 `[13,13,13,13,13,13,13,9]`。只对它们可使用 `original_manifests` 原样执行快照collector中的 `--episode-manifest --trust-episode-manifest` 分支。旧7任务的null noise不满足该分支的int转换要求，不能直接作为固定noise重采manifest执行。

## 在另一台机器复现

1. 复制本目录，运行 `sha256sum -c SHA256SUMS`。
2. 安装相同RoboTwin/OpenPI代码、仿真资源、robot/camera配置、base权重及归一化assets。协议包不包含这些大文件。当前collector快照中的 `ROBOTWIN_ROOT` 和采集Slurm中的 `PI05_ROOT` 等须替换成本机路径；确认base loader实际加载的是同一份权重。
3. 如果要求数据完全一致，先迁移原始数据（以下命令在目标机器执行，目标路径必须是新目录）：

```bash
rsync -aL --files-from=rsync_source_files.txt \
  USER@SOURCE_HOST:<EVO_RL_ROOT>/ \
  /YOUR/NEW/source_tree/
```

`-L`会解引用原staging中的symlink，确保得到真实文件而不是跨机器失效的链接。对照 `data_transfer_manifest.json` 将源路径中的evo-RL前缀替换为 `/YOUR/NEW/source_tree/`；如需重排目录，按各文件destination映射。成功率应从迁移的meta校验，不应要求新仿真运行强行产生相同结果。
4. 若只要固定协议重新采集：新增3任务用原manifest/noise；旧7任务若另定固定noise必须另存新版本，并标明不是原700条数据。不得覆盖本包或原训练数据。
5. 如要重建训练replay，沿用全局episode_id、split和bootstrap masks，不重新randperm划分。

本次仅导出协议，没有修改训练数据、训练任务或重新启动采集。
