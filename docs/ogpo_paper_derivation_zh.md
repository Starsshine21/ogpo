# U-DIVL Flash-OGPO for PI0.5 and DexJoCo：算法推导与论文写作参考

## 0. 文档定位

本文档描述当前仓库中面向 PI0.5 VLA 和 DexJoCo 的离线生成式策略优化方法，目标是为论文的方法章节、算法框图、消融设计和实现细节提供参考。文档严格区分三类内容：

1. **问题定义与理论目标**：希望优化的双层 MDP 和离线 RL 目标。
2. **当前代码实现**：仓库中已经实现并由测试覆盖的数学形式。
3. **待实验验证的研究主张**：可以作为论文假设，但不能在正式对照实验完成前写成结论。

当前主方法名称为：

> **U-DIVL Flash-OGPO**
> Uncertainty-Aware Distributional Implicit Value Learning
> + Conservative Flash Offline Generative Policy Optimization

本文档不把 selected-transition Flash 目标称为完整 GSPO sequence-ratio，也不把当前 smoke 或短训练结果表述为性能提升证据。

---

## 1. 方法概览

整体系统包含两个不同时间尺度的 MDP。

### 1.1 外层：机器人 action-chunk MDP

外层状态由视觉、本体状态和语言组成：

$$
s_t=(I_t^{base},I_t^{wrist},p_t,ell).
$$

PI0.5 一次生成长度为 $H$ 的 action chunk：

$$
A_t=(a_{t,0},a_{t,1},\ldots,a_{t,H-1})\in\mathbb R^{H\times D}.
$$

当前 `click_mouse` 配置中：

$$
H=30,\qquad D=22.
$$

环境采用 receding-horizon 执行，只执行前 $m$ 步后重新规划。当前 replay 构建配置使用：

$$
m=4.
$$

因此外层 critic 的动作语义不是完整生成 chunk，而是实际产生状态转移的执行前缀：

$$
Q(s_t,A_{t,0:m}).
$$

### 1.2 内层：PI0.5 flow 生成 MDP

PI0.5 从高斯噪声 $x_1$ 出发，通过 $K$ 个 flow Euler transition 生成 clean action endpoint $x_0$。内层状态是 flow latent $x_{t_k}$，内层动作是一次 stochastic flow transition：

$$
x_{t_{k+1}}\sim p_\theta(\cdot\mid x_{t_k},s,t_k).
$$

外层 critic 只评价最终 endpoint：

$$
Q(s,A(x_0)),
$$

而不学习：

$$
Q(s,x_t,t).
$$

当前实现也不通过 $\nabla_A Q(s,A)$ 更新 PI0.5。Actor 使用 detached endpoint advantage 和 flow-transition score-function/PPO ratio 更新。

### 1.3 为什么仍属于 offline RL

训练阶段满足：

1. 状态 $s$ 只来自冻结 replay $\mathcal D$。
2. critic reward 和 next state 只来自 $\mathcal D$。
3. 当前策略生成的候选 action 只用于 actor extraction。
4. 候选 action 不进入 critic Bellman max backup。
5. DexJoCo 只用于独立 evaluation，evaluation reward 不写回 replay。

因此 actor 虽然在内层 flow MDP 上进行 on-policy 或 near-on-policy 更新，整体机器人学习过程仍是 offline RL。

---

## 2. 符号表

| 符号 | 含义 |
|---|---|
| $s_t$ | 外层机器人状态，包含图像、本体和语言 |
| $A_t$ | PI0.5 生成的完整 action chunk |
| $H$ | 生成 chunk 长度，当前为 30 |
| $D$ | 单步环境 action 维度，当前为 22 |
| $m_t$ | 实际执行 prefix 长度，当前通常为 4 |
| $M_t$ | execution mask |
| $R_t^{(m)}$ | 执行前缀的折扣 chunk return |
| $Q_{\phi_j}$ | 第 $j$ 个 action-chunk scalar critic |
| $Z_{\psi_j}(s)$ | replay action target-Q 的状态条件 categorical distribution |
| $V_j(s)$ | 从 $Z_j(s)$ 提取的 adaptive quantile baseline |
| $G$ | 每个 replay state 的候选 action 数量 |
| $x_t$ | PI0.5 normalized flow action latent |
| $v_\theta$ | PI0.5 velocity field |
| $K$ | flow Euler steps，正式配置为 10 |
| $\pi_\theta$ | 当前 actor |
| $\pi_{old}$ | rollout 和 old log-prob 策略 |
| $\pi_{ref}$ | 冻结 SFT/BC reference policy |
| $\rho$ | PPO importance ratio |

---

## 3. DexJoCo 动作空间与 PI0.5 边界

### 3.1 环境动作不是 arm-joint delta

当前单臂 `click_mouse` 策略输出 22 维动作：

$$
a=[p_{eef}^{xyz},r_{eef}^{rotvec},q_{hand}],
$$

其中：

$$
p_{eef}^{xyz}\in\mathbb R^3,\quad
r_{eef}^{rotvec}\in\mathbb R^3,\quad
q_{hand}\in\mathbb R^{16}.
$$

它的控制语义为：

- 机械臂：EEF Cartesian **absolute pose**；
- 手部：16 维 **absolute joint position**；
- 姿态在 policy 边界使用 rotation vector，执行前转换为 quaternion；
- 没有把输出与当前状态相加，因此不是 delta action。

传给 DexJoCo 底层环境的 23 维格式为：

$$
[p_{eef}^{xyz},q_{eef}^{quat},q_{hand}].
$$

### 3.2 三个动作空间

实现中需要区分三个空间：

1. **DexJoCo raw action space**：22 维 rotvec 表示，用于 replay、critic 和 metrics。
2. **PI0.5 normalized flow space**：经过 checkpoint input normalization，用于 FM 和 PPO transition。
3. **DexJoCo execution space**：rotvec 转 quaternion 后的 23 维底层 action。

设 checkpoint 输入和输出变换分别为 $T_{in}$ 与 $T_{out}$，则：

$$
\widetilde A=T_{in}(A),\qquad A=T_{out}(\widetilde A).
$$

FM 和 stochastic flow transition 位于 $\widetilde A$ 空间；critic 必须评价 $T_{out}(x_0)$，不能直接评价 normalized latent。

---

## 4. 固定离线 action-chunk replay

### 4.1 正确的 chunk transition

对第 $t$ 次规划，实际执行长度为 $m_t$。定义 mask：

$$
M_{t,h}=\mathbf 1[h<m_t],\qquad h=0,\ldots,H-1.
$$

有效 critic action 为：

$$
\overline A_t=M_t\odot A_t.
$$

未执行 suffix 显式置零。因此对任意只修改 suffix 的扰动 $\Delta A$：

$$
M_t\odot(A_t+\Delta A)=M_t\odot A_t,
\quad \text{if }M_t\odot\Delta A=0.
$$

这保证未执行 action 不会被错误归因为 $s_{t+m}$ 的原因。

### 4.2 Chunk return 与折扣

执行前缀的 reward 为：

$$
R_t^{(m_t)}=\sum_{j=0}^{m_t-1}\gamma^j r_{t+j}.
$$

跨 chunk bootstrap discount 为：

$$
d_t=\gamma^{m_t}.
$$

因此 replay transition 是：

$$
(s_t,\overline A_t,R_t^{(m_t)},d_t,s_{t+m_t},done_t).
$$

### 4.3 数据划分

当前实现按 episode 划分 train、validation 和 held-out，避免同一条 trajectory 同时出现在 critic 训练和 uncertainty calibration 中。

Success buffer 的定义是成功 episode 中的所有 chunks：

$$
\mathcal D_{success}=\{(s,A)\in\mathcal D:\text{episode-success}=1\}.
$$

只有真实 terminal chunk 携带 terminal reward；Monte Carlo return 将成功信号向 episode 前部传播：

$$
G_t=\sum_{k=t}^{T-1}\gamma^{k-t}r_k.
$$

---

## 5. PI0.5 velocity flow 与随机 transition

### 5.1 原生 PI0.5 flow matching

OpenPI 的时间约定是：

$$
t=1\text{ 为噪声},\qquad t=0\text{ 为 clean action}.
$$

给定 normalized clean action $A$ 和高斯噪声：

$$
\epsilon\sim\mathcal N(0,I),
$$

训练插值为：

$$
x_t=t\epsilon+(1-t)A.
$$

沿该直线路径对 $t$ 求导：

$$
\frac{\partial x_t}{\partial t}=\epsilon-A.
$$

因此 velocity target 为：

$$
u_t=\epsilon-A.
$$

标准 flow-matching loss：

$$
\mathcal L_{FM}
=\mathbb E_{(s,A)\sim\mathcal D,t,\epsilon}
\left[
\|v_\theta(s,x_t,t)-(\epsilon-A)\|_2^2
\right].
$$

当前时间采样近似：

$$
t\sim 0.999\cdot\operatorname{Beta}(1.5,1)+0.001.
$$

### 5.2 确定性 Euler sampler

连续 ODE 写为：

$$
\frac{dx_t}{dt}=v_\theta(s,x_t,t).
$$

推理从 $x_1=\epsilon$ 向 $t=0$ 反向积分。令：

$$
\Delta t=-\frac{1}{K},
$$

则 Euler 更新为：

$$
x_{k+1}=x_k+\Delta t\,v_\theta(s,x_k,t_k).
$$

### 5.3 ODE-to-SDE adapter

原生 PI0.5 inference 不提供 transition log-prob。为了构造 PPO ratio，当前实现将 Euler mean 包装成对角高斯 transition：

$$
\mu_\theta(x_k,s,t_k)
=x_k+\Delta t\,v_\theta(s,x_k,t_k),
$$

$$
p_\theta(x_{k+1}\mid x_k,s,t_k)
=\mathcal N(\mu_\theta,\operatorname{diag}(\sigma_\theta^2)).
$$

配置中的 `stochastic_variance` 是初始化方差 $\nu$：

$$
\sigma_0=\sqrt{\nu},\qquad \log\sigma_0=\frac12\log\nu.
$$

对 latent 维度 $d=1,\ldots,HD$ 求和，transition log-prob 为：

$$
\log p_\theta(x'\mid x)
=-\frac12\sum_d
\left[
\frac{(x'_d-\mu_{\theta,d})^2}{\sigma_{\theta,d}^2}
+2\log\sigma_{\theta,d}
+\log(2\pi)
\right].
$$

两个对角高斯 transition 的 KL 为：

$$
D_{KL}(p_\theta\|p_{ref})
=\sum_d
\left[
\log\frac{\sigma_{ref,d}}{\sigma_{\theta,d}}
+\frac{\sigma_{\theta,d}^2+(\mu_{\theta,d}-\mu_{ref,d})^2}
{2\sigma_{ref,d}^2}
-\frac12
\right].
$$

**论文表述边界**：这个 stochastic transition 是 OGPO 训练 adapter 的建模选择，不是 PI0.5 原生生成分布的精确概率模型。

---

## 6. PI0.5 JAX 全量 actor 参数化

当前 actor 使用原生 JAX PI0.5，完整参数集合为：

$$
\theta=
\{\theta_{PI0.5},\theta_{residual},\log\sigma\}.
$$

PI0.5 base velocity 不再 stop-gradient：

$$
v_{base}=f_{\theta_{PI0.5}}(s,x_t,t).
$$

Residual MLP 的输入是当前 latent、base velocity 和 timestep：

$$
r_\theta=r_\theta([x_t,v_{base},t]).
$$

最终 velocity：

$$
v_\theta=v_{base}+r_\theta.
$$

Residual 最后一层以零初始化，因此训练开始时：

$$
r_\theta=0,\qquad v_\theta=v_{base}.
$$

这使初始 policy 与 SFT checkpoint 对齐。训练使用 rematerialization 与
Adafactor 控制显存；完整参数保存到 Orbax sidecar，而不是只保存 compact
residual。

三套策略为：

$$
\pi_\theta,\qquad \pi_{old},\qquad \pi_{ref}.
$$

- $\pi_\theta$：当前全量 trainable JAX PI0.5；
- $\pi_{old}$：完整 actor 的 EMA，用于 rollout 和 old log-prob；
- $\pi_{ref}$：冻结原始 SFT PI0.5，永不更新。

---

## 7. 外层 scalar Q ensemble

### 7.1 Critic 输入

每个 member 独立学习：

$$
Q_{\phi_j}(s,\overline A),\qquad j=1,\ldots,M.
$$

当前生产 critic 使用共享 Gemma3+SigLIP 多模态状态 backbone。两路 RGB 由
全量可训练 SigLIP 编码，视觉 token、语言 token、本体 token 和 readout token
进入全量可训练 Gemma3；动作执行 prefix 由 masked temporal action attention 编码。
三组 Q-V pair 只在 head 处分叉。完整公式见第 25 节。

每个 Q member 有独立 head 和对应 target copy。可选 randomized prior：

$$
Q_j(s,A)=Q_j^{train}(s,A)+\beta_{prior}Q_j^{prior}(s,A),
$$

其中 prior 参数冻结。

因此论文中应将当前版本描述为共享 VLM backbone 的多模态 action-chunk
critic，而不是纯 replay-state MLP critic。它与 PI0.5 使用同类
Gemma/SigLIP 组件，但并不共享同一个运行时参数对象。

### 7.2 Bootstrap diversity

对每个 batch 和 ensemble member 独立采样：

$$
b_{j,n}\sim\operatorname{Bernoulli}(p_{boot}).
$$

第 $j$ 个 member 的 Q loss 只在 $b_{j,n}=1$ 的样本上计算，从而增加 ensemble 数据视角差异。

---

## 8. U-DIVL：状态条件 replay-value distribution

### 8.1 DIVL 的精确定义

对每个 critic member，定义：

$$
Z_{\psi_j}(s).
$$

它表示在固定状态 $s$ 下，从 replay behavior action distribution 取动作时，对应 target-Q 值的 categorical distribution：

$$
A\sim\mu_{\mathcal D}(A\mid s),
\qquad Q_{\bar\phi_j}(s,A)\sim Z_j(s).
$$

它不是：

1. 指定动作的 return distribution $Z(s,A)$；
2. Q ensemble 本身；
3. 候选动作的 epistemic uncertainty；
4. flow-state critic $Q(s,x_t,t)$。

### 8.2 Categorical support

设固定 support：

$$
\mathcal Z=\{z_i=v_{min}+i\Delta z\}_{i=0}^{N-1},
$$

$$
\Delta z=\frac{v_{max}-v_{min}}{N-1}.
$$

自动 support 是保留的消融选项。若启用，它根据冻结 replay MC-return 范围一次性生成；若 replay 范围为 $[G_{min},G_{max}]$，margin fraction 为 $c$：

$$
v_{min}=G_{min}-c(G_{max}-G_{min}),
$$

$$
v_{max}=G_{max}+c(G_{max}-G_{min}).
$$

当前正式配置与 LWD 一致，直接使用：

$$
[v_{min},v_{max}]=[-0.1,1.1],\qquad N=201.
$$

Support 已纳入 checkpoint，避免 actor 端使用不同 atom 坐标解释同一组 DIVL logits。

### 8.3 单点 categorical projection

对 target scalar $y$，先裁剪：

$$
\hat y=\operatorname{clip}(y,v_{min},v_{max}).
$$

连续 atom index：

$$
b=\frac{\hat y-v_{min}}{\Delta z},
\qquad l=\lfloor b\rfloor,
\qquad u=\lceil b\rceil.
$$

投影概率：

$$
p_l=u-b,\qquad p_u=b-l.
$$

当 $l=u$ 时实现通过 scatter-add 保证总概率仍为 1。最后再次归一化以避免浮点误差。

对 replay action：

$$
y_{j,n}^{DIVL}=Q_{\bar\phi_j}(s_n,A_n^{data}),
$$

$$
\widehat Z_{j,n}=\operatorname{Proj}_{\mathcal Z}
(y_{j,n}^{DIVL}).
$$

DIVL 交叉熵 loss：

$$
\mathcal L_{DIVL}
=-\frac{1}{MB}\sum_{j,n,i}
\widehat Z_{j,n,i}\log Z_{\psi_j,i}(s_n).
$$

### 8.4 Entropy-adaptive quantile

归一化 entropy：

$$
u_j(s)=
\frac{-\sum_i p_{j,i}(s)\log p_{j,i}(s)}{\log N}
\in[0,1].
$$

当前正式配置采用 LWD offline schedule：

$$
\alpha_j(s)=
\operatorname{clip}(0.6-0.3u_j(s),0.5,0.6).
$$

因此：

- $u\to0$：distribution 集中，使用较高 quantile；
- $u\to1$：distribution 弥散，quantile 回落到 $\alpha_{min}$。

定义 CDF：

$$
F_j(z_i\mid s)=\sum_{k=0}^{i}p_{j,k}(s).
$$

找到最小 $i$ 满足：

$$
F_j(z_i\mid s)\ge\alpha_j(s),
$$

直接选择该离散 atom，不在相邻 support atoms 间插值：

$$
V_j(s)=\operatorname{Quantile}
(Z_j(s),\alpha_j(s)).
$$

这就是候选动作必须超过的 replay-supported absolute value threshold。

---

## 9. Critic target 推导

### 9.1 DIVL bootstrap

默认 next-state value 使用 ensemble mean，而不是 minimum：

$$
V_{DIVL}(s')=\frac1M\sum_{j=1}^{M}V_j^{target}(s').
$$

原因是 sparse-success 数据中 ensemble minimum 可能过度悲观，阻碍成功信号传播。Minimum target 保留为消融。

### 9.2 Reference-policy variance reduction

可选地从冻结 reference policy 采样：

$$
A'_r\sim\pi_{ref}(\cdot\mid s'),
\qquad r=1,\ldots,N_{vr}.
$$

计算：

$$
V_{ref}(s')=
\frac{1}{N_{vr}M}\sum_{r,j}
Q_{\bar\phi_j}(s',A'_r).
$$

混合 bootstrap：

$$
V_{boot}(s')=
\lambda_{divl}V_{DIVL}(s')
+(1-\lambda_{divl})V_{ref}(s').
$$

该功能默认关闭，以控制 PI0.5 sampling 成本。

### 9.3 TD 与 Monte Carlo mixture

Chunk TD target：

$$
y_t^{TD}=R_t^{(m_t)}
+\gamma^{m_t}(1-done_t)V_{boot}(s_{t+m_t}).
$$

可选 MC mixture：

$$
y_t=(1-\lambda_{MC})y_t^{TD}
+\lambda_{MC}G_t.
$$

### 9.4 Critic objective

带 bootstrap mask 的 Q loss：

$$
\mathcal L_Q=
\frac{
\sum_{j,n}b_{j,n}
(Q_{\phi_j}(s_n,A_n)-y_n)^2
}{
\sum_{j,n}b_{j,n}
}.
$$

总 critic loss：

$$
\mathcal L_{critic}
=\mathcal L_Q+\lambda_Z\mathcal L_{DIVL}.
$$

Target network 使用 Polyak update：

$$
\bar\phi\leftarrow(1-\tau)\bar\phi+\tau\phi,
$$

$$
\bar\psi\leftarrow(1-\tau)\bar\psi+\tau\psi.
$$

---

## 10. Critic calibration 与 uncertainty

### 10.1 Ensemble epistemic uncertainty

对候选动作：

$$
\mu_Q(s,A)=\frac1M\sum_jQ_j(s,A),
$$

$$
\sigma_Q(s,A)=
\sqrt{\frac1M\sum_j(Q_j(s,A)-\mu_Q)^2}.
$$

这里的 $\sigma_Q$ 是候选动作 epistemic proxy，与 DIVL entropy 的状态级 replay ambiguity 含义不同。

### 10.2 Conformal scaling

在 validation set 上计算 nonconformity score：

$$
r_n=
\frac{|\mu_Q(s_n,A_n)-G_n|}
{\max(\sigma_Q(s_n,A_n),\epsilon)}.
$$

取经验 $(1-\delta)$ quantile：

$$
c_\delta=\operatorname{Quantile}_{1-\delta}({r_n}).
$$

校准后 uncertainty：

$$
\widetilde\sigma_Q=c_\delta\sigma_Q.
$$

目标 coverage 为：

$$
\Pr
\left(
|\mu_Q-G|\le\widetilde\sigma_Q
\right)\approx1-\delta.
$$

当前实现同时记录 RMSE、Huber error、rank correlation、pairwise ranking accuracy、coverage、ECE、disagreement-error correlation 和 Q exploitation gap。

### 10.3 Pairwise ranking accuracy

对一个固定 validation batch 中的任意两个不同样本 $i<j$，先用三个 critic
的均值作为预测分数：

$$
\bar Q_i=\frac{1}{M}\sum_{m=1}^{M}Q_m(s_i,A_i).
$$

监督目标优先使用 Monte Carlo return $G_i$；若 replay 没有 MC return，则
退化为 chunk return。定义：

$$
\Delta^Q_{ij}=\bar Q_i-\bar Q_j,\qquad
\Delta^G_{ij}=G_i-G_j.
$$

忽略 $\Delta^G_{ij}=0$ 的平局 pair，pairwise ranking accuracy 为：

$$
\operatorname{Acc}_{\mathrm{pair}}
=
\frac{
\sum_{i<j}\mathbb 1[\Delta^G_{ij}\ne0]\,
\mathbb 1[
\operatorname{sign}(\Delta^Q_{ij})
=
\operatorname{sign}(\Delta^G_{ij})
]
}{
\sum_{i<j}\mathbb 1[\Delta^G_{ij}\ne0]
}.
$$

不依赖公式渲染器的等价写法：

```text
q_i = mean(Q_1(s_i, A_i), Q_2(s_i, A_i), Q_3(s_i, A_i))
q_j = mean(Q_1(s_j, A_j), Q_2(s_j, A_j), Q_3(s_j, A_j))

predicted_order = sign(q_i - q_j)
target_order    = sign(MC_return_i - MC_return_j)

pair_correct = 1  if predicted_order == target_order
pair_correct = 0  otherwise

ranking_accuracy = correct_non_tie_pairs / all_non_tie_pairs
```

它只检查 critic 是否把“回报更高的 action chunk”排在“回报更低的 action
chunk”前面，不要求 Q 的绝对数值已经校准。随机二分类排序的期望约为
$0.5$，所以旧 critic 的 $0.5218$ 只能说明略高于随机，不能作为可靠 actor
verifier。

当前 100-episode 实验有三种使用方式：

1. critic 训练期间每 100 steps 从固定 validation replay 采样 32 条，最多
   形成 $\binom{32}{2}=496$ 个 pair，用于阶段诊断和阶段切换；
2. critic 结束后的 calibration 使用完整 validation replay 计算该指标，
   Slurm 要求其不低于 $0.55$，否则 actor 依赖不释放；
3. actor 运行时复用一个固定 seed 采出的 32 条 validation 子集做启动门控，
   避免把完整视觉 validation replay 一次送入 GPU。该运行时检查是附加保护，
   正式准入仍以完整 validation replay 的 calibration 为准。

---

## 11. U-DIVL conservative advantage

### 11.1 候选动作

对每个 replay state，用 old policy 生成 $G$ 个候选：

$$
A_i\sim\pi_{old}(\cdot\mid s),
\qquad i=1,\ldots,G.
$$

所有 candidate endpoint 在 `torch.no_grad()` 下通过 critic：

$$
q_{i,j}=Q_j(s,A_i).
$$

Actor backward 后显式检查 critic 和 reference 参数 gradient 为 `None`。

### 11.2 Absolute DIVL advantage

每个 member 的 absolute advantage：

$$
\Delta_{i,j}=Q_j(s,A_i)-V_j(s).
$$

与 group mean baseline 不同，$V_j(s)$ 来自 replay action value distribution。因此即使某个候选是当前组中最好的，只要它仍低于 replay quantile，advantage 就不会被错误变成正值。

### 11.3 Two-sided sign consensus

定义正负 margin $m_+,m_-$。保守 advantage 为：

$$
A_i^{cons}=
\begin{cases}
\min_j\Delta_{i,j},
&\min_j\Delta_{i,j}>m_+,\\
\max_j\Delta_{i,j},
&\max_j\Delta_{i,j}<-m_-,\\
0,&\text{otherwise}.
\end{cases}
$$

解释：

1. 全体 critic 同意更好：取最保守的正幅度；
2. 全体 critic 同意更差：取最保守的负幅度；
3. critic 对符号有分歧：不更新该候选。

### 11.4 Running MAD normalization

不用每个 state group 内的标准差，而使用跨 batch 的 running MAD：

$$
\operatorname{MAD}(A)=
1.4826\cdot
\operatorname{median}
\left(|A-\operatorname{median}(A)|\right).
$$

忽略 consensus 产生的大量零值，然后进行 EMA：

$$
s_t=\beta s_{t-1}+(1-\beta)\operatorname{MAD}(A_t).
$$

归一化：

$$
\widehat A_i=
\operatorname{clip}
\left(
\frac{A_i^{cons}}{s_t+\epsilon},
-A_{max},A_{max}
\right).
$$

可选 warm-up mixture：

$$
A_i^{mix}=\lambda_{abs}\widehat A_i
+(1-\lambda_{abs})A_i^{group}.
$$

默认 $\lambda_{abs}=1$。Group-normalized GRPO advantage 只作为消融。

### 11.5 LCB 消融

校准 LCB：

$$
Q_{LCB}(s,A_i)=
\mu_Q(s,A_i)-\kappa\widetilde\sigma_Q(s,A_i),
$$

$$
A_i^{LCB}=Q_{LCB}(s,A_i)
-\frac1M\sum_jV_j(s).
$$

它保留为 uncertainty ablation，不替代主方法 sign consensus。

---

## 12. 状态不确定性与 behavior support gate

### 12.1 DIVL entropy state gate

状态级 entropy：

$$
u_{state}(s)=\frac1M\sum_ju_j(s).
$$

状态权重：

$$
w_{state}(s)=
\exp(-\eta_Hu_{state}(s)).
$$

可选 Adaptive PPO clip：

$$
\epsilon(s)=\epsilon_{min}
+(1-u_{state}(s))
(\epsilon_{max}-\epsilon_{min}).
$$

可选 Adaptive reference KL coefficient：

$$
\beta_{KL}(s)=
\beta_0(1+c_{KL}u_{state}(s)).
$$

生产默认只保留第三个作用，即高 entropy 状态获得更低 advantage weight：

```yaml
uncertainty:
  adapt_ppo_clip: false
  adapt_kl_beta: false
```

此时

$$
\epsilon(s)=\epsilon_{\max},\qquad
\beta_{KL}(s)=\beta_0,\qquad
w_{state}(s)=\exp(-\eta_Hu_{state}(s)).
$$

分别打开两个开关后，才恢复上面的 adaptive clip 或 adaptive KL 公式。这样
避免同一个 $u_{state}$ 同时通过 advantage、clip 和 KL 三条路径重复收紧
actor；同时保留独立消融能力。Full 的生产 `ais_joint` 模式不读取逐状态
$\epsilon(s)$，始终使用固定的 `actor.ppo_clip_chain`。

### 12.2 Behavior support distance

当前实现使用候选与 paired replay action 的 RMS 距离：

$$
d_{support}(s,A_i)=
\sqrt{
\frac1{HD}\|A_i-A_{data}\|_2^2
}.
$$

这里比较的是 raw action space 中的完整 $H$ 步 chunk；它与 critic 的
execution-mask 语义不同。前者是当前实现采用的行为支持启发式，后者才决定
Bellman transition 中哪些动作真正造成了 next state。论文中不应把该 RMS
距离解释为严格的 behavior likelihood 或仅执行前缀距离。

结合 ensemble uncertainty：

$$
w_{support}(s,A_i)=
\exp
\left(
-\lambda_{epi}\widetilde\sigma_Q(s,A_i)
-\lambda_{sup}d_{support}(s,A_i)
\right).
$$

若：

$$
d_{support}>d_{max},
$$

则硬置零。

最终 actor advantage：

$$
A_i^{final}=
w_{state}(s)
w_{support}(s,A_i)
A_i^{mix}.
$$

如果 state entropy 超阈值或候选 consensus ratio 低于阈值，则该 state 的所有候选 advantage 置零。

---

## 13. Full-Chain OGPO

### 13.1 Rollout buffer

Old policy 保存完整 stochastic flow chain：

$$
\{x_k,x_{k+1},t_k,
\log p_{old}(x_{k+1}\mid x_k,s,t_k)}_{k=0}^{K-1}.
$$

同一个 endpoint advantage $A_i^{final}$ 广播到该 trajectory 的所有 transition。

### 13.2 Importance ratio

逐 transition ratio：

$$
\rho_{i,k}=
\exp
\left[
\log p_\theta(x_{i,k+1}\mid x_{i,k},s,t_k)
-\log p_{old}(x_{i,k+1}\mid x_{i,k},s,t_k)
\right].
$$

实现先裁剪 log-ratio：

$$
\log\rho\leftarrow
\operatorname{clip}(\log\rho,-20,20),
$$

再指数化，避免 overflow。

### 13.3 Full PPO objective

$$
\mathcal L_{Full}
=-\mathbb E_{i,k}
\left[
\min
\left(
\rho_{i,k}A_i^{final},
\operatorname{clip}
(\rho_{i,k},1-\epsilon(s),1+\epsilon(s))
A_i^{final}
\right)
\right].
$$

Full 的主要优点是 credit 广播覆盖完整 flow chain；主要缺点是所有 $K$ 个 transition 都要重新计算 log-prob，激活和反传成本高。

---

## 14. Flash-OGPO

### 14.1 Iso-temporal grouping

对每个 replay state $s_b$ 只采样一个 selected step：

$$
k_b\sim\operatorname{Uniform}\{0,\ldots,K-1\}.
$$

同一 state 的 $G$ 个候选共享 $k_b$，但初始噪声不同。这保证组内候选在相同 flow 时间尺度上比较。

### 14.2 单随机 transition

除 selected step 外，所有 transition 使用 deterministic Euler mean：

$$
x_{k+1}=\mu_{old}(x_k,s,t_k),
\qquad k\ne k_b.
$$

Selected step 使用随机 transition：

$$
x_{k_b+1}\sim p_{old}
(\cdot\mid x_{k_b},s,t_{k_b}).
$$

只有该 step 保存 old log-prob，并在更新时重新计算 new log-prob。

### 14.3 Flash objective

$$
\rho_i^{Flash}=
\exp
\left[
\log p_\theta(x_{k_b+1}^i\mid x_{k_b}^i,s,t_{k_b})
-\log p_{old}(x_{k_b+1}^i\mid x_{k_b}^i,s,t_{k_b})
\right].
$$

$$
\mathcal L_{Flash}
=-\mathbb E_i
\left[
w_{rect}(k_b)
\min
\left(
\rho_i^{Flash}A_i^{final},
\operatorname{clip}
(\rho_i^{Flash},1-\epsilon(s),1+\epsilon(s))
A_i^{final}
\right)
\right].
$$

### 14.4 Temporal rectification

对当前 transition：

$$
\mu_\theta=x+\Delta t\,v_\theta,
$$

Gaussian score 对 velocity 的链式比例包含：

$$
\left|\frac{\partial\mu}{\partial v}\right|
\frac1\sigma
=\frac{|\Delta t|}{\sigma}.
$$

当前 $\Delta t$ 和 $\sigma$ 不随 timestep 变化，因此 analytic correction 在各 timestep 相同。归一化后：

$$
w_{rect}(k)=1.
$$

Empirical EMA 模式记录每个 timestep 的 raw gradient norm：

$$
g_k^{EMA}\leftarrow
\beta g_k^{EMA}+(1-\beta)g_k.
$$

权重：

$$
w_{rect}(k)=
\operatorname{clip}
\left(
\frac{\overline g^{EMA}}{g_k^{EMA}+\epsilon},
w_{min},w_{max}
\right).
$$

### 14.5 Full 与 Flash 的理论边界

如果 full objective 是对 $K$ 个 transition loss 的均值，并且从同一 stochastic trajectory 均匀抽一个 transition，则单步抽样可以构成 full mean 的 Monte Carlo estimator。

当前 Flash 为节省计算，采用“selected step 随机、其余 step deterministic”的不同 trajectory distribution。因此它应被描述为 Full-OGPO 的 compute-efficient selected-transition approximation，不能不加条件地声称为 Full objective 的严格无偏估计。

---

## 15. Reference、FM anchor 和 success regularization

### 15.1 Reference KL 与可选 state adaptation

对 transition：

$$
\mathcal L_{KL}(s)=
\beta_{KL}(s)
D_{KL}
\left(
p_\theta(\cdot\mid x,s,t)
\|p_{ref}(\cdot\mid x,s,t)
\right).
$$

它限制 policy 离开 SFT behavior support。生产默认
$\beta_{KL}(s)=\beta_0$；只有
`uncertainty.adapt_kl_beta: true` 时，DIVL entropy 较高的状态才使用更强
KL。

当前代码中的 KL 采样位置与两种 actor objective 对齐：Flash 在 selected
transition 上计算 KL；Full 为节省额外前向成本，只在保存链的第一个
transition 上计算 KL，而不是对全部 $K$ 个 transition 求和。因此，上式是
通用正则形式，当前 Full 实现对应其中的单 transition Monte Carlo/代理项。

### 15.2 Offline FM anchor

从独立 replay batch 采样真实 action，继续优化标准 PI0.5 flow matching：

$$
\mathcal L_{anchor}=\mathcal L_{FM}(\mathcal D).
$$

它不依赖 critic，作用是防止 policy 利用 Q 误差后丢失原始行为能力。

### 15.3 Success buffer

在成功 episode 的 chunks 上增加：

$$
\mathcal L_{success}
=\mathcal L_{FM}(\mathcal D_{success}).
$$

这使 sparse-success replay 中的成功行为获得更高 actor anchor 采样频率。

### 15.4 Action smoothness

对非 gripper 维度可选：

$$
\mathcal L_{smooth}
=\frac1{H-1}\sum_h
\|a_{h+1}-a_h\|_2^2
+\eta\frac1{H-2}\sum_h
\|a_{h+2}-2a_{h+1}+a_h\|_2^2.
$$

离散或特殊 gripper 维度通过 mask 排除。

### 15.5 总 actor loss

$$
\mathcal L_{actor}
=\mathcal L_{OGPO}
+\mathcal L_{KL}
+\lambda_{FM}\mathcal L_{anchor}
+\lambda_{success}\mathcal L_{success}
+\lambda_{smooth}\mathcal L_{smooth}.
$$

$\mathcal L_{OGPO}$ 根据方法选择 Full 或 Flash。

---

## 16. 为什么 actor 不需要 action-space Q gradient

候选 endpoint 及 advantage 在 `no_grad` 下计算：

$$
A_i^{final}=\operatorname{stopgrad}
\left[
\mathcal A(Q(s,A_i),Z(s))
\right].
$$

Actor 梯度来自 transition log-prob：

$$
\nabla_\theta\mathcal L_{OGPO}
\propto
-A_i^{final}
\nabla_\theta
\log p_\theta(x'\mid x,s,t),
$$

而不是：

$$
\nabla_AQ(s,A)\frac{\partial A}{\partial\theta}.
$$

因此 critic 可以是非光滑 ensemble、categorical baseline 或包含 hard gate，而不要求 Q 对 action 的梯度可用。

---

## 17. 完整训练 pipeline

### Phase A：PI0.5 baseline 与固定数据

1. 用原始 JAX/Orbax PI0.5 checkpoint 做 baseline rollout。
2. 保存 RGB、proprioception、language、generated chunk、executed action、reward、done 和 success。
3. 转成固定 torch chunk replay。
4. 按 episode 划分 train、validation、held-out。
5. 构建 success、failure 和可用时的 near-success buffer。

### Phase B：U-DIVL critic warm-up

重复：

1. 从固定 replay 采样 batch；
2. 计算 mask-aware Q；
3. 从 target DIVL 提取 next-state adaptive quantile；
4. 构造 chunk TD/MC target；
5. 更新 Q ensemble 和 DIVL；
6. Polyak 更新 target networks；
7. 在 validation 上记录 ranking、coverage、entropy 和 exploitation gap。

Critic 完成后拟合 conformal scale，并保存 calibrated checkpoint。

### Phase C：Full-Chain OGPO

1. 加载同一个 calibrated critic；
2. old policy 生成完整 stochastic flow trajectories；
3. critic 只评价 endpoint；
4. 构造 U-DIVL conservative advantage；
5. 对所有 transition 做 PPO update；
6. 加入 KL、FM 和 success loss；
7. 同步 old policy；
8. 独立做 DexJoCo evaluation。

### Phase D：Flash-OGPO

从同一个 calibrated critic 和同一个 SFT reference 独立启动：

1. 每个 replay state 采一个 selected timestep；
2. 除 selected step 外使用 deterministic rollout；
3. 只重算 selected transition log-prob；
4. 使用与 Full 完全相同的 endpoint critic 和 advantage；
5. 比较 compute、稳定性和 success rate。

### Phase E：安全停止

以下条件触发 actor skip 或 stop：

- ranking accuracy 或 coverage 未达到启动阈值；
- policy-reference KL 超阈值；
- candidate ensemble disagreement 超阈值；
- support distance 超阈值；
- actor loss 或 importance ratio 非有限；
- DIVL entropy 或 consensus 不满足 state gate。

---

## 18. 从 OGPO-style 方法迁移到 PI0.5 的关键改动

| 迁移问题 | 通用 OGPO-style 假设 | PI0.5/DexJoCo 实现 |
|---|---|---|
| 外层动作 | 单步连续动作或完整生成样本 | 实际执行 prefix 的 action chunk |
| 环境转移 | action 与 next state 一一对应 | 生成 30 步但只执行前 4 步，必须 mask suffix |
| 生成参数化 | 常见 diffusion noise/score | PI0.5 velocity flow，$t=1$ noise、$t=0$ action |
| Sampler | 已有 stochastic transition | 原生 Euler ODE 外包对角 Gaussian transition |
| 动作坐标 | 单一 normalized space | PI normalized flow 与 raw absolute EEF action 分离 |
| Policy 更新 | 全模型或小网络 | 原生 JAX PI0.5 全量更新，附加 zero-init residual 与 log-std |
| Critic | 评价最终 sample | 只评价 raw DexJoCo endpoint，不评价 flow latent |
| Offline 约束 | KL 或 behavior penalty | reference KL + FM anchor + success buffer + support gate |
| 计算成本 | 完整生成链 PPO | Full baseline + selected-transition Flash 主方法 |

迁移的核心不是把通用 diffusion PPO 公式直接套到 PI0.5，而是保持 PI0.5 的真实 interpolation、时间方向、Euler coefficient、normalization transform 和环境 action semantics。

---

## 19. 相对基础 OGPO-style 设计的改进点

### 19.1 U-DIVL absolute baseline

基础 group-relative advantage 只能判断“组内谁更好”，不能判断“是否优于 replay behavior”。U-DIVL 使用 replay target-Q distribution 的 adaptive quantile，提供 absolute support-aware threshold。

可作为论文核心假设：

> 在纯离线生成式策略优化中，absolute replay-value baseline 比纯组内 baseline 更能避免从整体较差的候选组中强化相对最好但仍然 OOD/低价值的动作。

### 19.2 Ensemble sign consensus

不是直接使用 ensemble mean，而要求所有 critic 对 advantage 符号一致。它同时限制正向强化和负向抑制，并将符号分歧样本置零。

### 19.3 两类 uncertainty 分工

- DIVL entropy：状态级 replay-value ambiguity；
- Q ensemble disagreement：候选动作 epistemic uncertainty。

二者不混为同一个 uncertainty，分别控制 state trust region 和 candidate support weight。

### 19.4 Conformal calibration

Raw ensemble std 不保证 coverage。Conformal scale 用 validation MC return 校准 uncertainty 幅度，使 LCB 和 support gate 更可解释。

### 19.5 Conservative Flash extraction

Flash 只在一个 transition 上保留 stochastic graph，降低完整 PI0.5 flow chain 的 activation 成本，同时保留 endpoint critic 信号。

### 19.6 多重 behavior anchor

Reference KL、全 replay FM anchor、success FM 和 action support distance 同时约束 policy，降低 frozen offline critic 被 OOD action 利用的风险。

### 19.7 显式训练 gate

Actor 不因训练脚本进入 Phase C/D 就自动更新，而是要求 critic ranking、coverage 和 entropy 合格；运行中还监控 KL、disagreement、support 和 finite statistics。

---

## 20. 论文算法伪代码

```text
Input:
    frozen replay D
    success buffer D_success
    frozen PI0.5 reference policy pi_ref
    Q/DIVL ensemble size M

Phase 1: U-DIVL critic
    initialize {Q_j, Z_j, Qbar_j, Zbar_j}_{j=1..M}
    repeat critic_steps:
        sample chunk transitions from D
        mask every unexecuted action suffix
        compute Vbar_j(s') from entropy-adaptive quantile of Zbar_j(s')
        compute mean-ensemble chunk TD target
        optionally mix TD target with MC return
        update each Q_j with independent bootstrap mask
        project Qbar_j(s, A_data) to categorical support
        update Z_j by cross entropy
        soft-update target networks
    fit conformal scale on validation trajectories
    freeze calibrated critic for actor scoring

Phase 2: Full or Flash actor extraction
    initialize policy residual at zero
    set old_policy <- policy
    keep reference_policy frozen
    repeat actor_steps:
        sample replay states, FM batch, and success batch
        optionally update critic only from fixed replay
        sample G flow trajectories per replay state using old_policy
        map clean normalized endpoints back to raw DexJoCo action space
        score endpoints with frozen Q ensemble under no_grad
        obtain DIVL adaptive-quantile baselines
        compute two-sided sign-consensus advantage
        normalize by running MAD
        apply state entropy and behavior-support gates

        if Full:
            recompute log-prob for every stochastic flow transition
            optimize full-chain clipped PPO objective
        if Flash:
            sample one shared timestep per state group
            recompute only the selected transition log-prob
            optimize temporally rectified clipped PPO objective

        add reference KL, replay FM, success FM, and optional smoothness
        assert critic/reference gradients are absent
        clip actor gradient and update residual
        synchronize old_policy by hard copy or EMA
        stop when safety guard is violated
```

---

## 21. 建议的论文实验矩阵

### 21.1 主对照

1. PI0.5 SFT/BC；
2. scalar single-Q + AWR；
3. scalar Q ensemble + Full-OGPO；
4. DIVL + Full-OGPO；
5. DIVL + Flash-OGPO；
6. U-DIVL + Flash-OGPO。

### 21.2 关键消融

1. ensemble mean vs sign consensus；
2. adaptive quantile vs fixed quantile；
3. DIVL absolute baseline vs group-normalized GRPO；
4. no entropy state gate；
5. raw ensemble std vs conformal uncertainty；
6. no reference KL；
7. no FM anchor；
8. no success buffer；
9. no behavior support weight；
10. Full vs Flash；
11. analytic vs empirical temporal rectification；
12. mean target vs minimum target；
13. mask vs no executed-action mask。

### 21.3 推荐报告指标

环境指标：

- success rate，使用相同 seeds 和相同 replanning ratio；
- average return 和 episode length；
- action smoothness、gripper error 和 constraint violation；
- task-wise success。

Critic 指标：

- validation Q RMSE；
- pairwise ranking accuracy；
- rank correlation；
- conformal coverage；
- ensemble disagreement-error correlation；
- DIVL entropy 和 saturation；
- predicted-Q vs realized-return exploitation gap。

Actor 指标：

- positive/negative/zero consensus ratio；
- policy-reference KL；
- support distance；
- importance ratio 和 clip fraction；
- FM/success loss；
- selected-timestep gradient norm；
- Full/Flash wall-clock、显存和 samples/s。

---

## 22. 可写入论文与暂不能写成结论的内容

### 22.1 当前可作为方法事实描述

- 双层外层 chunk-MDP 与内层 flow-MDP 的定义；
- critic 不学习 $Q(s,x_t,t)$；
- actor 不使用 $\nabla_AQ$；
- mask-aware chunk transition；
- U-DIVL categorical replay-value distribution；
- adaptive quantile、sign consensus、conformal uncertainty；
- Full 和 Flash 两种 actor objective；
- reference KL、FM anchor 和 success buffer；
- 原生 JAX PI0.5 全参数 actor，使用 Adafactor 与 rematerialization；
- 训练与环境 evaluation 分离。

### 22.2 需要正式实验后才能写成结论

- U-DIVL 显著提高 success rate；
- Flash 与 Full 达到相同或更高最终性能；
- conformal calibration 必然减少 Q exploitation；
- sign consensus 在所有任务上优于 LCB；
- entropy gate 对多任务泛化有效；
- 当前方法优于已有 Q-gradient 或 QAM 方法。

### 22.3 当前实现限制

1. Critic 已使用共享 Gemma3+SigLIP 多模态 backbone；本地真实权重 GPU
   smoke `784418` 已通过，正式 critic `784431` 已完成训练和 conformal
   校准。
2. Actor 已改为原生 JAX PI0.5 全量微调；PyTorch residual-only 路径只保留
   为历史 baseline。
3. ODE-to-SDE variance 是训练 adapter 的建模假设。
4. Paired replay-action RMS 只是简化 support proxy，不等同于精确 behavior likelihood。
5. 当前 replay 规模较小，validation uncertainty 估计可能高方差。
6. Flash deterministic-except-selected trajectory 与 Full stochastic trajectory 分布不同。
7. 多 GPU/DDP 和 PI prefix cache 尚未实现。
8. JAX raw-action smoothness 尚未可微实现，生产默认关闭。
9. 真实 2.3B PI0.5 的 Adafactor backward、critic-derived PPO 非零梯度、
   fused EMA、Orbax sidecar 和恢复推理已分别由 GPU Jobs `786571`、
   `786596`、`786582` 验证；但正式 100-step 训练和环境 success-rate
   evaluation 尚未完成，不能据此声称算法性能提升。

---

## 23. 代码与公式对应关系

| 数学模块 | 实现文件 |
|---|---|
| Chunk mask、return、discount | `dexjoco/dexjoco/ogpo/chunk_transition.py` |
| Replay split 和 success buffer | `dexjoco/dexjoco/ogpo/replay.py` |
| Zarr 到 chunk replay | `dexjoco/dexjoco/ogpo/zarr_replay.py` |
| Scalar Q ensemble | `dexjoco/dexjoco/ogpo/critic.py` |
| Gemma3+SigLIP state encoder | `dexjoco/dexjoco/ogpo/gemma_siglip_backbone.py` |
| 三组 Q-V heads 与 masked action pool | `dexjoco/dexjoco/ogpo/multimodal_critic.py` |
| Subsample-min target | `dexjoco/dexjoco/ogpo/critic_targets.py` |
| Categorical support、projection、quantile | `dexjoco/dexjoco/ogpo/distributional_value.py` |
| DIVL target 和 adaptive quantile | `dexjoco/dexjoco/ogpo/divl.py` |
| Sign consensus、LCB、MAD | `dexjoco/dexjoco/ogpo/conservative_advantage.py` |
| Conformal、entropy gate、support weight | `dexjoco/dexjoco/ogpo/uncertainty.py` |
| PI0.5 interpolation、Euler、Gaussian transition | `dexjoco/dexjoco/ogpo/openpi_flow_spec.py` |
| Gaussian log-prob 和 KL | `dexjoco/dexjoco/ogpo/flow_logprob.py` |
| 历史 PyTorch residual adapter | `dexjoco/dexjoco/ogpo/pi05_pytorch_adapter.py` |
| JAX PI0.5 全量 actor、Adafactor、Orbax sidecar | `dexjoco/dexjoco/ogpo/pi05_jax_adapter.py` |
| JAX rollout、PPO、KL、FM | `dexjoco/dexjoco/ogpo/pi05_jax_flow_core.py` |
| Full PPO objective | `dexjoco/dexjoco/ogpo/full_ogpo.py` |
| Flash rollout 和 objective | `dexjoco/dexjoco/ogpo/flash_ogpo.py` |
| Temporal rectification | `dexjoco/dexjoco/ogpo/temporal_rectification.py` |
| FM、success、smoothness | `dexjoco/dexjoco/ogpo/losses.py` |
| 总训练流程和 checkpoint | `dexjoco/dexjoco/ogpo/trainer.py` |
| Offline calibration metrics | `dexjoco/dexjoco/ogpo/evaluator.py` |

---

## 24. 一句话方法摘要

> U-DIVL Flash-OGPO first learns an uncertainty-calibrated distributional value threshold over behavior-supported action chunks in the outer offline robot MDP, then conservatively full-finetunes a native JAX PI0.5 flow policy through sign-consensus endpoint advantages and selected-transition PPO updates, while preserving behavior support using reference KL, replay flow matching, and successful-trajectory anchors.

对应中文：

> U-DIVL Flash-OGPO 首先在外层离线机器人 action-chunk MDP 中学习经过不确定性校准的 behavior-supported distributional value threshold，然后利用 ensemble sign-consensus endpoint advantage 和 selected-transition PPO 保守优化冻结 backbone 的 PI0.5 flow policy，并通过 reference KL、离线 flow matching 与成功轨迹 anchor 维持行为支持。

---

## 25. Gemma3+SigLIP 三头 U-DIVL critic 推导

### 25.1 共享多模态状态表示

对相机 $c\in\{base,wrist\}$，全量可训练 SigLIP 产生视觉 token：

$$
U_c=\operatorname{SigLIP}(I_c)\in\mathbb R^{L_c\times d_v},\qquad
\widehat U_c=U_cW_v\in\mathbb R^{L_c\times d_g}.
$$

语言 token 和本体 token 为：

$$
E_\ell=\operatorname{Embed}_{Gemma}(\ell),\qquad e_p=W_pp+b_p.
$$

拼接可学习 readout token $e_r$：

$$
X=[\widehat U_{base};\widehat U_{wrist};E_\ell;e_p;e_r],
$$

共享 Gemma3 得到状态表示：

$$
h_s=\operatorname{Gemma3}(X)_{readout}\in\mathbb R^{d_g}.
$$

三个 Q/V heads 共享同一个 Gemma3+SigLIP 参数对象。当前 `full_td` 阶段对
SigLIP、Gemma3、视觉投影、本体投影、readout token、action encoder 和全部
Q/V heads 一起反向传播，不再安装 LoRA。

### 25.2 Masked temporal action encoder

只允许已执行前缀进入 Q。对第 $h$ 步：

$$
u_h=W_a\frac{a_h-\mu_a}{\sigma_a}+b_a+e_h.
$$

令 $M_h=1$ 表示该步实际执行，以学习 query $q_a$ 做 masked attention：

$$
h_A=\operatorname{LN}\left(
\operatorname{MHA}(q_a,\{u_h:M_h=1\},\{u_h:M_h=1\})
\right).
$$

因此，对任意只改变 suffix 的 $\delta A$，若 $M\odot\delta A=0$，则：

$$
h_A(A+\delta A,M)=h_A(A,M),\qquad Q_m(s,A+\delta A)=Q_m(s,A).
$$

### 25.3 三组对应 Q-V heads

共享 $h_s,h_A$，只让末端 heads 独立：

$$
Q_m(s,A)=f_m^Q([h_s;h_A]),\qquad
z_m(s)=f_m^V(h_s)\in\mathbb R^C,quad m=1,2,3.
$$

类别概率与 entropy 为：

$$
p_{m,c}(s)=\frac{e^{z_{m,c}}}{\sum_{c'}e^{z_{m,c'}}},\qquad
H_m(s)=-\sum_c p_{m,c}\log p_{m,c}.
$$

在 support $\{\zeta_c\}_{c=1}^{C}$ 上按 entropy 选择分位数 $\alpha(s)$：

$$
V_m(s)=F^{-1}_{Z_m(s)}(\alpha(s)).
$$

这里的“三头”不是共享最后一个 scalar 输出，而是三组独立 $Q_m$ 与 $Z_m/V_m$，使 two-sided 判断保留 pair-level 不确定性。

### 25.4 三头 mean DIVL target

保留用户指定的三个 Q/V pairs。为了避免 three-head minimum 与
two-sided actor consensus 重复施加悲观偏置，TD bootstrap 使用三个 target
value quantile 的均值：

$$
V_{boot}(s')=\frac{1}{3}\sum_{m=1}^{3}V_m^{-}(s').
$$

$$
y=R^{(m)}+\gamma^m(1-done)V_{boot}(s').
$$

当前 `click_mouse` 使用 1-step chunk TD、`gamma=0.9999`，不再混合
reference-policy sampled Q 或 Monte Carlo return。

Q loss 使用独立 bootstrap mask $b_{m,n}$：

$$
\mathcal L_Q=
\frac{\sum_{m,n}b_{m,n}(Q_m(s_n,A_n)-y_n)^2}
{\sum_{m,n}b_{m,n}}.
$$

Value distribution 用对应 target-Q 的 categorical projection $\Phi$：

$$
\mathcal L_Z=-\frac1{3B}\sum_{m,n,c}
\Phi(Q_m^{-}(s_n,A_n))_c\log p_{m,c}(s_n).
$$

总 critic loss 为：

$$
\mathcal L_{critic}=\mathcal L_Q+\lambda_Z\mathcal L_Z.
$$

### 25.5 Two-sided absolute conservative advantage

候选 $A_j$ 对每个 pair 的绝对 advantage：

$$
\Delta_{m,j}=Q_m(s,A_j)-V_m(s).
$$

$$
A_j^{CA}=\begin{cases}
\min_m\Delta_{m,j},&\Delta_{m,j}>\delta_+,\ \forall m,\\
\max_m\Delta_{m,j},&\Delta_{m,j}<-\delta_-,\ \forall m,\\
0,&\text{otherwise}.
\end{cases}
$$

这保留 OGPO+CA 的双侧符号共识，但 baseline 从候选组均值改成 U-DIVL replay-value threshold。生产不除以每组标准差，只使用非零 advantage 的跨 batch Running MAD：

$$
\widetilde A_j=\operatorname{clip}\left(
\frac{A_j^{CA}}{\operatorname{EMA-MAD}(A^{CA})+\epsilon},-c,c
\right).
$$

### 25.6 Full-chain AIS PPO

对完整 flow chain $\tau=(x_K,\ldots,x_0)$：

$$
\log\rho_\tau=\sum_{k=1}^{K}
[\log\pi_\theta(x_{k-1}|s,x_k)-\log\pi_{EMA}(x_{k-1}|s,x_k)],
$$

$$
\rho_\tau=\exp(\operatorname{clip}(\log\rho_\tau,-c_\rho,c_\rho)).
$$

clip 只在 chain level 做一次：

$$
\mathcal L_{AIS}=-\mathbb E\left[
\min(\rho_\tau\widetilde A,
\operatorname{clip}(\rho_\tau,1-\epsilon_K,1+\epsilon_K)\widetilde A)
\right].
$$

正式初值 $\epsilon_K=0.01$。逐 transition PPO 仅保留为显式消融。

### 25.7 PI0.5 reverse-time corrected SDE

OGPO 原式使用 cleanward 时间 $\tau:0\to1$：

$$
\nabla_x\log p_\tau(x)=\frac{\tau v_\tau-x}{1-\tau},\qquad
\sigma(\tau)=\sigma_0\sqrt{1-\tau}.
$$

因此修正项中的 $(1-\tau)$ 解析消去：

$$
c_\tau=\frac{\sigma_0^2}{2}(\tau v_\tau-x).
$$

PI0.5 使用 $t=1-\tau$ 且 $v_{PI}=dx/dt=-v_\tau$。先变换连续时间公式，再离散化，得到：

$$
\widetilde v_{PI}=v_{PI}+\frac{\sigma_0^2}{2}[(1-t)v_{PI}+x_t],
$$

$$
\mu_{t-\Delta t}=x_t-\Delta t\,\widetilde v_{PI},\qquad
\sigma_{PI}(t)=\sigma_0\sqrt t.
$$

噪声在 $t=1$ 最大，在 clean endpoint $t=0$ 为零。采样、log-prob 重算和 KL 必须使用完全相同的 $\mu$ 与 $\sigma(t)$。

## 26. 方法来源与边界

| 机制 | 来源定位 | 当前状态 |
|---|---|---|
| Categorical DIVL、EMA-Q 投影、entropy-adaptive quantile、quantile TD bootstrap | LWD DIVL 核心 | 保留数学主干 |
| Gemma3+SigLIP、共享 VLM readout、temporal action pooling、独立 Q/V heads | LWD critic 架构 | 借鉴并扩展为三组 Q-V heads |
| 联合链 ratio、corrected SDE、EMA actor、success buffer、two-sided CA | OGPO/OGPO+ 核心或稳定化机制 | 保留并实现 |
| 三组 Q-V、bootstrap mask、three-head mean bootstrap、conformal/support/ranking gates | 本项目 U-DIVL 扩展 | 实现 |
| Flash selected-transition PPO | 本项目对 OGPO 的计算压缩 | 当前正式 actor |
| Candidate group-relative baseline | OGPO 默认 | 仅消融，非生产默认 |
| Group std normalization | GRPO 常见做法，OGPO 原文也不采用 | 非生产默认 |
| Critic VLM 全量微调 | 可选工程策略 | 因小 replay 弃用 |
| Best-of-N inference | OGPO 可选推理 trick | 仅消融，正式 `N=1` |

### 26.1 当前 critic 与 LWD DIVL 的一致部分

二者都执行以下数据流：

```text
replay (s, A, r, s')
  -> target Q(s, A) 投影到 categorical value support
  -> 学习 state-conditioned replay-Q distribution V(s)
  -> 按 V(s') 的 entropy 选择 quantile
  -> 用 quantile(V(s')) 构造 chunk TD target
  -> 更新 Q(s, A)
  -> EMA/Polyak 更新 target network
```

因此当前方法可以准确描述为“以 LWD DIVL 为 value-learning 主干”。

### 26.2 当前 critic 不是 LWD 原样复刻

| 项目 | LWD 原文 | 当前实现 |
|---|---|---|
| Q/V 数量 | clipped double-Q，两组 scalar Q，共享一个 categorical V 设计 | 三组对应的 `Q_m` 与 `V_m` |
| target 聚合 | 两个 Q 取全局 minimum | 三个 value quantile 取均值，避免叠加悲观偏置 |
| DIVL target | clipped target-Q 投影 | 每个 `V_m` 投影对应的 EMA `Q_m` |
| TD bootstrap | 100% DIVL quantile | 100% DIVL quantile |
| 冷启动 | short task 1-step、long task 10-step TD | `click_mouse` 直接使用 1-step chunk TD |
| support | 201 atoms，固定 `[-0.1, 1.1]` | 201 atoms，固定 `[-0.1, 1.1]` |
| discount | 原文实验 `gamma=0.9999` | `gamma=0.9999` |
| Q loss | squared TD loss | MSE |
| critic 参数更新 | offline 阶段全量微调 value/critic VLM | Gemma3+SigLIP 共享 backbone 全量微调 |
| optimizer | Adam `5e-4` + cosine，基于 652.5 小时多任务 replay | Adam `5e-4` + cosine；从第 1 step 起全量微调共享 Gemma3+SigLIP |
| 数据循环 | offline buffer 后接持续 online mixed replay | 当前 100 episodes 构建一次固定 replay，尚未自动循环部署采集 |
| 额外稳定器 | 不以 ranking/conformal gate 为核心 | bootstrap ensemble、ranking、coverage、conformal 和 support gate |

### 26.3 Actor 与 LWD 完全不同

LWD 使用 QAM：

```text
计算 action gradient dQ/dA
  -> 解 adjoint dynamics
  -> 把结果变成 flow velocity regression target
```

当前 actor 使用 OGPO/Flash-PPO：

```text
critic 只输出 detached advantage
  -> 不计算 dQ/dA
  -> 用 stochastic flow transition likelihood ratio 做 PPO
  -> 加 reference KL、FM anchor 和 success BC
  -> 对 JAX PI0.5 全参数求梯度
```

所以不能把整体算法称为 LWD，也不能称 critic 为 LWD 的严格复现。准确表述应为：

> Critic 以 LWD DIVL 为数学主干，并加入三头 ensemble、保守聚合和校准门控；
> actor 则采用 OGPO 风格的 likelihood-ratio policy extraction，而不是 LWD
> 的 QAM。
