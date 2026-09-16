# DexJoCo PI0.5 JAX Full-Finetune OGPO 实现与进展

更新时间：2026-07-27

本文是当前 OGPO 工程实现、状态和限制的唯一权威说明。论文公式与方法来源见
`docs/ogpo_paper_derivation_zh.md`，实际命令见 `docs/ogpo_runbook_zh.md`。

## 1. 当前目标

当前路线不再以“冻结 PI0.5 + PyTorch residual”为最终目标。正式目标是：

1. 用 PyTorch 训练和校准共享 Gemma3+SigLIP 的三组 Q-V U-DIVL critic；
2. critic 在外层 action-chunk MDP 为候选 endpoint 提供 detached advantage；
3. actor 保持原生 JAX/Orbax PI0.5，OGPO loss 通过
   `jax.value_and_grad` 更新完整 PI0.5、residual 和 transition log-std；
4. 使用 Adafactor 和 rematerialization 将全量微调放入单张 A100-40G；
5. 保存并恢复 current actor、EMA old actor 和 Optax state；
6. 推理加载全量微调后的 JAX actor，而不是回退到原始 SFT 参数。

Critic 和 actor 不跨框架反向传播。PyTorch critic 只输出 detached advantage
与诊断；PI0.5 的策略梯度、KL 和行为正则全部在 JAX 内计算。

## 2. 整体 Pipeline

### 2.1 外层 chunk MDP

Replay transition 为：

$$
(s_t,A_t,M_t,R_t^{(m)},\gamma^m,s_{t+m},d_t).
$$

其中 $A_t$ 是生成 action chunk，$M_t$ 只保留真实执行 prefix。Critic 不学习
$Q(s,x_\tau,\tau)$，也不向 actor 传递 $\nabla_AQ$。

### 2.2 Critic

生产 critic 使用：

- Gemma3-270M 与 SigLIP2-So400M/14@224 全量微调；
- 一个共享多模态 backbone；
- 三组独立 $Q_m(s,A)$ 和 categorical $Z_m(s)$ heads；
- execution-mask aware temporal action encoder；
- 三个 value heads 分位数均值的 TD bootstrap；
- LWD 1-step DIVL target、MSE、`gamma=0.9999`；
- 201 atoms，固定 support `[-0.1,1.1]`；
- conformal calibration。

三组 absolute advantage：

$$
\Delta_{m,j}=Q_m(s,A_j)-V_m(s),
$$

通过 two-sided sign consensus 得到保守 advantage。生产默认不使用
group-relative baseline，也不做组内标准差归一化。

### 2.3 JAX PI0.5 Actor

`PI05JaxActorModule` 包含：

$$
\theta=
\{\theta_{\mathrm{PI0.5}},\theta_{\mathrm{residual}},\log\sigma\}.
$$

`nnx.split(actor)` 将完整 PI0.5 backend 纳入 `actor_state`。更新链路为：

```text
jax.value_and_grad(loss)(actor_state)
  -> nnx.merge(graphdef, actor_state)
  -> PI0.5 predict_velocity
  -> PPO + KL + FM/success gradients
  -> optax.adafactor
  -> new actor_state
```

`predict_velocity` 不包含 `stop_gradient`。`train=False` 只关闭图像增强和
dropout，确保 PPO transition likelihood 可重复，不会冻结参数。

Residual 最后一层为零初始化，因此初始 JAX actor 与原始 PI0.5 一致。

### 2.4 一次 Flash actor update 到底做什么

以下是当前 `100ep` 正式配置的一次 outer step；这一节使用纯文本表达，避免
Markdown 数学渲染器差异：

```text
输入:
  1 个 replay state s
  old actor      = PI0.5 的 EMA 副本
  reference actor = 永久冻结的 SFT PI0.5
  critic          = 三组 Q_m / V_m

1. 从 10 个 flow steps 中均匀选一个 k。
2. old actor 从噪声生成 1 条 action chunk A。
   除第 k 步外走确定性 flow；第 k 步保留随机 transition (x_k -> x_{k-1})。
3. 把最终 A 转回 DexJoCo raw action，并交给三个 Q heads 评分。
4. 对每个 head 计算 delta_m = Q_m(s, A) - V_m(s)。
5. three-head two-sided consensus:
     三个 delta 都 > 0: advantage = 最小的正 delta；
     三个 delta 都 < 0: advantage = 最接近 0 的负 delta；
     符号有分歧:       advantage = 0。
6. advantage 经 running-MAD、w_state 和 support weight 缩放。
7. 在同一个 JAX trace 中重算 old/current 对第 k 个 transition 的 log-prob。
8. ratio = exp(logp_current - logp_old)，代入 clipped PPO loss。
9. 再加入:
     frozen-SFT reference KL
     replay flow-matching anchor
     success-buffer flow-matching
10. jax.value_and_grad 对完整 PI0.5 参数求梯度。
11. Adafactor 更新完整 PI0.5、residual head 和 log-std。
12. 用 EMA 同步 old actor；每 10 个 outer steps 保存 checkpoint。
```

这里最容易混淆的边界是：

- critic 不参与 JAX 反向传播；
- 不计算 `dQ/dA`；
- critic 只把候选 action chunk 变成 detached scalar advantage；
- 真正给 PI0.5 梯度的是 PPO log-prob、reference KL 和两个
  flow-matching loss；
- 正 advantage 提高这条生成 transition 的概率，负 advantage 降低概率，
  disagreement advantage 为 0 时 PPO 部分不更新。

当前 `batch_size=1` 表示每个 outer step 取一个 replay state；
`group_size=1` 表示该 state 只生成一个候选 action chunk。Two-sided
consensus 仍然由三个 critic heads 完成，但不再有同状态候选组内比较。

## 3. Actor Loss

### 3.1 PPO

Flash 在 selected transition 上使用：

$$
\mathcal L_{\mathrm{PPO}}=
-\mathbb E\left[
w_\tau
\min\left(
\rho A,
\operatorname{clip}(\rho,1-\epsilon,1+\epsilon)A
\right)
\right].
$$

Full `ais_joint` 使用整条链的联合 ratio：

$$
\rho_{\mathrm{chain}}=
\exp\left(
\sum_k\log\pi_\theta(x_{k-1}|x_k,s)
-\log\pi_{\mathrm{old}}(x_{k-1}|x_k,s)
\right).
$$

### 3.2 U-State

状态 uncertainty 继续影响 advantage：

$$
w_{\mathrm{state}}(s)=\exp(-\eta_Hu_{\mathrm{state}}(s)).
$$

生产默认：

```yaml
uncertainty:
  adapt_ppo_clip: false
  adapt_kl_beta: false
```

因此：

$$
\epsilon(s)=\epsilon_{\max},\qquad
\beta_{\mathrm{KL}}(s)=\beta_0.
$$

只有显式开启 `adapt_kl_beta` 时才使用：

$$
\beta_{\mathrm{KL}}(s)=
\beta_0(1+c_{\mathrm{KL}}u_{\mathrm{state}}(s)).
$$

JAX 与 PyTorch actor 路径现在遵循同一公式。默认关闭 adaptation 不等于关闭
KL。

### 3.3 Reference KL

冻结 SFT reference 与 EMA old actor 职责不同：

- reference policy：行为约束和 KL；
- old policy：rollout 与 PPO likelihood ratio。

JAX KL 在对应 transition 的对角高斯分布之间计算，并保留固定
`regularization.beta_kl`。

### 3.4 FM 与 Success Buffer

FM anchor 在 JAX loss 内计算：

$$
x_t=t\epsilon+(1-t)A,\qquad
u_t=\epsilon-A,
$$

$$
\mathcal L_{\mathrm{FM}}=
\|v_\theta(s,x_t,t)-u_t\|_2^2.
$$

Success buffer 使用相同 JAX flow-matching loss，只采样成功轨迹。两项都对
完整 PI0.5 参数产生梯度，不再以 detached PyTorch 常数加入。

当前 JAX 路径对 `lambda_smooth != 0` 明确报错，因为 raw-action output
transform 尚未实现可微 JAX 版本；生产配置默认 `lambda_smooth: 0`，不会静默
丢失该梯度。

## 4. 显存策略

全量 PI0.5 约 2.3B 参数。当前使用：

- bfloat16 模型参数；
- `jax.checkpoint(..., nothing_saveable)` rematerialization；
- `torch.cuda.empty_cache()` 后进入 actor `value_and_grad`；
- Adafactor factorized second moment；
- current/reference/old 初始共享不可变 JAX arrays；
- JAX 与 PyTorch critic 共卡时限制 JAX memory fraction。

AdamW 仍可显式选择，但不作为 A100-40G 默认。

## 5. Checkpoint 与推理

主 checkpoint：

```text
<checkpoint>.pt
```

保存 critic、配置、运行状态和轻量 adapter 镜像。JAX 全量状态保存到 Orbax
sidecar：

```text
<checkpoint>.pt.jax/
```

Sidecar 包含：

- current `policy_actor_state`；
- EMA `old_policy_actor_state`；
- Adafactor `actor_opt_state`。

Resume 必须同时保留 `.pt` 文件和 `.pt.jax/` 目录。

推理检测 `pi05_jax_full_finetune` 格式后，从原始 SFT Orbax checkpoint 构建
模型结构，再覆盖 sidecar 中的 finetuned current actor。Reference policy
仍保持原始 SFT 参数。

## 6. 已完成验证

### Critic

- GPU smoke `784418`：真实 Gemma/SigLIP 权重、LoRA backward、
  suffix-invariance、checkpoint reload 通过；
- 正式 critic `784431`：三阶段训练完成，在 step 2700 early stop；
- calibrated checkpoint：
  `outputs/ogpo/click_mouse_gemma_udivl_calibrated.pt`；
- calibration：Q RMSE `0.132582`，pairwise ranking `0.626485`，
  interval coverage `0.898917`。

### JAX Actor 历史 Smoke

| Job | 结果 | 结论 |
|---|---|---|
| `785387` | FAILED | JAX SigLIP 收到 NCHW；已修复为 NHWC |
| `785908` | FAILED | `ml_dtypes.bfloat16` 无法直接转 Torch；已先升 float32 |
| `785909` | FAILED | critic 通过，AdamW actor backward OOM |
| `785917` | FAILED | remat 后仍被 AdamW state 挤占显存 |

Adafactor 端到端 GPU smoke 已于 2026-07-27 提交：

| Job | 配置 | 状态 | 验证目标 |
|---|---|---|---|
| `786557` | `configs/ogpo/pi05_jax_flash_ogpo_gpu_smoke.yaml` | FAILED | PI0.5 全参 backward 已通过；Adafactor 生成梯度平方临时量时再申请 2.25 GiB OOM |
| `786571` | 同上，fused optimizer step | COMPLETED | `actor_loss=0.0228`；全参 update 和 11 GiB Orbax sidecar 保存成功 |
| `786576` | finetuned checkpoint restore + inference | FAILED | inference 入口未安装 Orbax/JAX `record_scalar` 兼容 shim；已修复 |
| `786578` | 修复后的 restore + inference | FAILED | base+sidecar 恢复成功；验证脚本将巨型 bf16 leaf 升 fp32，额外申请 4.50 GiB OOM |
| `786580` | bounded-memory restore + inference | FAILED | backend 参数变化已确认；发现在线 `base/wrist` 与 critic replay 相机键未映射 |
| `786582` | camera-mapped restore + inference | COMPLETED | `(30,22)` action、reference divergence、critic Q 全部通过 |

日志：

```text
outputs/ogpo/logs/pi05-jax-smoke-786557.out
outputs/ogpo/logs/pi05-jax-smoke-786557.err
outputs/ogpo/logs/pi05-jax-smoke-786571.out
outputs/ogpo/logs/pi05-jax-smoke-786571.err
outputs/ogpo/logs/pi05-jax-restore-786576.out
outputs/ogpo/logs/pi05-jax-restore-786576.err
outputs/ogpo/logs/pi05-jax-restore-786578.out
outputs/ogpo/logs/pi05-jax-restore-786578.err
outputs/ogpo/logs/pi05-jax-restore-786580.out
outputs/ogpo/logs/pi05-jax-restore-786580.err
outputs/ogpo/logs/pi05-jax-restore-786582.out
outputs/ogpo/logs/pi05-jax-restore-786582.err
```

该轮证明 critic -> detached advantage -> JAX PPO/FM -> PI0.5 全参梯度已经接通。
失败发生在 backward 之后的 optimizer update。当前修复将
`clip + Adafactor + apply_updates` 合并为一个 JIT step，并 donate gradient 和
旧 optimizer-state buffer，避免同时常驻 grads、updates 和 new params。

`786571` 在 A100-40G 上于 7 分 18 秒完成，主 checkpoint 为 2.7 GiB，
JAX sidecar 为 11 GiB。该结果已经打通 critic update -> PI0.5 全参
backward -> Adafactor update -> checkpoint save。

`786582` 在 A100-40G 上于 1 分 37 秒完成真实恢复与推理：

```text
changed_backend_leaf=.../Transformer/encoder_norm/bias
sample_max_abs_difference=3.0517578e-05
action_shape=(30, 22)
policy_reference_action_divergence=6.6834326e-06
predicted_q=-0.036078759
```

`786571` 的单样本 two-sided consensus advantage 恰好为零，因此该轮参数更新
来自 FM anchor。为单独验证 critic -> PPO -> PI0.5 梯度，另设诊断配置
`pi05_jax_flash_ogpo_critic_signal_smoke.yaml`：临时使用 LCB advantage，并将
KL/FM/success 全部置零。该配置不改变生产默认的 two-sided sign consensus。
诊断 Job `786584` 的日志为
`outputs/ogpo/logs/pi05-jax-smoke-786584.{out,err}`。
该轮 critic-driven PPO backward 与 Adafactor update 已通过，但 eager EMA
old-policy 同步产生 2.25 GiB 临时量而 OOM。EMA 已改为融合 JIT；同时修复
`ema=0` 只同步 adapter、遗漏完整 PI0.5 backend 的问题。

`786589` 在 fused EMA 后完成，但隔离 metrics 显示初始 PPO ratio 为 `0.025`，
负 advantage 落入常数 clipped 分支，故 `actor_grad_norm=0`。根因是 rollout
scan 内与 standalone backward 中的 bf16 velocity 融合顺序不同，660 维联合
log-prob 放大了微小数值差。现在 old/current log-prob 在同一个 JAX loss trace
中重算，old 分支使用 `stop_gradient`；Flash 与 Full-chain 均采用该路径。
修复后复验 Job `786596` 已完成（6 分 25 秒）：

```text
importance_ratio_mean=1.0
conservative_advantage_abs_mean=0.0815001
flash_ppo_loss=0.0741816
actor_grad_norm=760.6247
fm_anchor_loss=0
reference_kl_beta=0
success_buffer_loss=0
```

因此该轮非零 PI0.5 梯度只能来自 critic-derived PPO；随后 fused EMA、
Adafactor 和 checkpoint 保存均完成。

正式作业提交前另用
`configs/ogpo/pi05_jax_flash_ogpo_regularization_smoke.yaml` 验证
PPO + fixed KL + FM + success 同时启用时的 A100-40G 峰值显存。Job
`786597` 已完成（8 分 03 秒）：

```text
importance_ratio_mean=1.0
fm_anchor_loss=0.2280408
success_buffer_loss=0.7326888
actor_grad_norm=2.1275
```

该轮随后完成 fused EMA 和 `.pt + .jax/` checkpoint 保存。

在线推理现在会使用 `flow.image_mapping` 的逆映射，把 DexJoCo client 的
`base/wrist` 图像填入 critic 所需的 `image_base/image_wrist`，不要求 client
重复发送同一图像的两套键。

### 本地测试

2026-07-27：

```text
119 passed
```

覆盖：

- JAX backend 参数得到非零更新；
- residual 零初始化；
- 固定和自适应 KL；
- FM/success 在 PPO advantage 为零时仍更新 backend；
- current/old/Adafactor Orbax roundtrip；
- trainer `.pt + .jax/` save/load；
- torch-facing rollout 使用当前 JAX actor state；
- JAX Flash 与 Full 最小更新。

## 7. 当前进度

| 项目 | 状态 |
|---|---|
| PyTorch U-DIVL critic 训练/校准 | 完成 |
| JAX PI0.5 全参数梯度链路 | 完成 |
| Adafactor/remat 低显存实现 | A100-40G GPU smoke 完成 |
| JAX PPO 与固定 KL 语义 | 完成 |
| JAX FM/success 梯度 | 完成 |
| 全量 actor/old/optimizer checkpoint | 完成；真实 sidecar 保存与 current actor 恢复通过 |
| 推理恢复 finetuned JAX actor | 完成，Job `786582` |
| 单步真实 PI0.5 actor backward/update/save | 完成，Job `786571` |
| 单步 critic-derived PPO 非零梯度 | 完成，Job `786596` |
| 正式全量 actor 训练 | 旧 Job `786720` 因 critic 门控全部跳过而停止；500-step 新链路 Job `787099` 已提交 |
| 多 seed DexJoCo 对照 | 未开始 |

本次要求的 critic 训练 checkpoint -> PI0.5 JAX 全量 actor update -> 可恢复推理
工程路径已完成。科学实验已完成 smoke，正式 critic/actor 依赖链已提交。

### 正式训练任务

2026-07-27 已提交：

| Job | 任务 | 时限 | 当前状态 |
|---|---|---|---|
| `786719` | Gemma3+SigLIP 三组 Q-V U-DIVL critic 训练与 conformal calibration | 4 小时 | COMPLETED，耗时 19 分 43 秒 |
| `786720` | PI0.5 JAX 100-step Flash-OGPO 全量微调 | 23 小时 30 分 | CANCELLED，运行 30 分 14 秒，无有效 actor update |

Actor 只有在 critic 正常完成并写出 calibrated checkpoint 后才会启动。
本轮 critic 的校准 `pairwise_ranking_accuracy=0.521791`，低于配置门槛
`0.55`。截至 actor step 6，metrics 中 `actor_skipped=1`，停止原因均为
`critic_ranking_accuracy_below_min`。所以依赖链已成功启动，但本轮尚不能
视为有效 PI0.5 全量微调实验。

针对该结果，新增 100-episode 数据与训练链：

| Job | 任务 | 配置 | 依赖 |
|---|---|---|---|
| `786734` | 使用原生 JAX PI0.5 采集 100 个 `click_mouse` episode | 采集完成，59% success；仅 replay 转换阶段失败 | 无 |
| `787097` | 从 `786734` 的完整 Zarr 补建 replay | r8cpu，正确的 `torch+zarr` 环境 | 无 |
| `787098` | 重新训练 Gemma3+SigLIP 三组 Q-V critic | VLM 全量微调，15000 steps，batch 8，LWD-aligned DIVL | `afterok:787097` |
| `787099` | PI0.5 JAX 全量 Flash-OGPO | 500 actor steps，batch 1，Adafactor | `afterok:787098` |

首次采集 `786728` 因 OpenPI 标准服务入口缺失 Orbax/JAX `record_scalar`
兼容初始化而失败；新增独立兼容服务包装器后，`786734` 已确认进入
`Episode 1/100`。

旧等待任务 `786735/786736`、`786758/786759` 和磁盘配额耗尽期间失败的
replay 补建链均已取消，由上述新链路替代。Critic Job `787098` 在
calibration 后执行
`pairwise_ranking_accuracy >= 0.55` 的 Slurm 硬门控。门控失败时 Job 返回
非零状态，actor 不会启动；actor 自身原有质量门控继续保留。

Actor 的 500 steps 对 10 个均匀采样 flow timestep 提供约 50 次/位置的期望
更新。由于本轮 actor 没有带来新的 online replay，actor 阶段冻结已完成
15000 steps 训练的 critic（`steps_per_actor_step=0`）；critic 继续参与
three-head two-sided advantage 和门控，只是不再对同一固定数据重复 TD 更新。

本轮不再使用旧 5-episode 稳定化配置中的 MC warm-up、Huber、
reference-action target mixture、51-atom auto support 或 Gemma LoRA。三个
Q/V heads 共享同一个全量可训练 Gemma3+SigLIP backbone，仅 prediction heads
独立。

全量 VLM GPU 验证：

```text
Job 786740: 1 optimizer step, batch 8, COMPLETED, 77 秒
Job 786752: 20 optimizer steps, batch 8, COMPLETED, 87 秒
loss: 8.3046 -> 2.9879
```

两者包含相同的模型加载、checkpoint 保存和完整旧 validation replay 校准。
按耗时差估算净训练约 0.53 秒/step，15000 steps 约 2.2 小时，正式 critic
的 4 小时时限保留。
日志：

```text
outputs/ogpo/logs/pi05-gemma-udivl-786719.{out,err}
outputs/ogpo/logs/pi05-flash-train-786720.{out,err}
```

## 8. 当前限制

1. 单步 smoke 总耗时 7 分 18 秒，包含模型加载、首次编译、训练和保存；正式训练
   的稳态 step 吞吐尚未测量；
2. 全量 Orbax sidecar 实测 11 GiB；同类 A100 节点恢复通过，跨拓扑恢复未验证；
3. 单 GPU 同时容纳 PyTorch critic 和 JAX actor，尚无 DDP/FSDP；
4. PI0.5 prefix/KV cache 未实现，每个 likelihood/FM forward 重算视觉语言
   prefix；
5. JAX raw-action smoothness 尚未实现，非零配置会明确失败；
6. 当前 replay 只有五个 episode，不能支持论文结论；
7. Best-of-N 默认关闭，正式推理为 `N=1`。

单张 A100-40G 的正式配置使用 `batch_size=1`、`group_size=1`、100 actor
steps，并每 10 步保存一次 checkpoint；Slurm 时限为 23 小时 30 分。更大的
同状态候选组需要 actor microbatching 或更大显存，不作为当前单卡默认。

## 9. 关键文件

| 功能 | 文件 |
|---|---|
| JAX actor、Adafactor、Orbax sidecar | `dexjoco/dexjoco/ogpo/pi05_jax_adapter.py` |
| JAX rollout、PPO、KL、FM | `dexjoco/dexjoco/ogpo/pi05_jax_flow_core.py` |
| Critic/actor 训练与 checkpoint | `dexjoco/dexjoco/ogpo/trainer.py` |
| JAX PI0.5 velocity API | `openpi/src/openpi/models/pi0.py` |
| 微调后推理加载 | `dexjoco/dexjoco/ogpo/inference_policy.py` |
| JAX production config | `configs/ogpo/pi05_jax_flash_ogpo.yaml` |
| JAX smoke config | `configs/ogpo/pi05_jax_flash_ogpo_gpu_smoke.yaml` |
| JAX smoke Slurm | `scripts/pi05_jax_flash_ogpo_gpu_smoke.slurm` |
| 真实 checkpoint 恢复/推理 smoke | `scripts/pi05_jax_ogpo_checkpoint_smoke.slurm` |
| JAX 路径测试 | `tests/ogpo/test_pi05_jax_actor_path.py` |
