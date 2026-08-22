# Footmimic 当前 MDP 总结：S1 mimic + S2 control

> **维护约定：本文件是当前 S1/S2 MDP 的唯一持续更新总表。** 后续 input、reward、termination、reset 或 diagnostics 发生变化时，直接修改 `MDP_SUMMARY.md`，不再新建 `MDP_SUMMARY_09.md` 等编号文件；旧的编号版中间总结已删除。

## 0.6 当前候选：Essay 16 shoulder/elbow residual v6

Essay 16 把上肢控制缩减为一个可辨识、无重复约束的接口：

1. S1 policy 输出仍为 29 维；S2 policy 输出从 29 维降为 23 维，即 15 个下肢/腰部 direct action 加 8 个肩肘 Gaussian pre-squash residual；
2. 8 个可学习关节为双侧 shoulder pitch/roll/yaw 与 elbow。6 个 wrist roll/pitch/yaw 不再占 policy 输出，直接执行每帧 live motion reference；
3. 肩肘只有一次物理变换：`q_target=q_ref+0.25*tanh(z)`，随后仅经过 joint soft limit 和已有 PD actuator；无 PCA、无 actor mean cap、无低通、无第二层 envelope；
4. 旧 post-tanh residual 正则替换为 `upper_body_pre_squash_action_l2`：raw=`mean(z_shoulder/elbow^2)`，weight `-0.15`、每步系数 `-0.003`。它在饱和区仍持续增大；当 `|z|=atanh(0.95)=1.8318` 时约贡献 `-0.0101/step`，与 v5 饱和成本预算相当，但不再存在 tanh 平坦区；
5. `action_rate_l2` 只对 8 个实际可学习 residual 计算变化率，reference-only wrist 和移动 reference 不计为策略抖动；
6. S1/S2 的 Actor/Critic 输入仍完全一致（163/295 维），`actions` observation 仍是完整 29 维最终 normalized joint target。变化的是 S2 actor 输出维度 23，不需要重训 S1，但旧 S2 checkpoint 必须经过一次输出头迁移；
7. scalar exploration std 的 `0.05` 有效域保护保留；deterministic play 保存 8 个 residual 的 raw/post-tanh telemetry，并在 14 维上肢数组中用 reference-only wrist 的零 residual/缺失 policy index 明确区分；ONNX 输出 23 维并只对 8 个 residual 维做 tanh。

V2--v5 checkpoint 不加 migration flag即可自动迁移到 v6：保留 shared/RNN、critic、normalizer 和旧 29 维输出中对应的 15 个下肢/腰部行；删除 6 个 wrist 行，重置 8 个肩肘行并把其 std 初始化为 `0.5`，同时丢弃不兼容的旧 optimizer。新 checkpoint 写入 `shoulder_elbow_reference_residual_v6`，之后正常 resume 时不得再带迁移参数。

## 0.5 Essay 15 历史候选：regularized direct residual v5

`diagnostic_20260822_015522.npz` 验证了 Essay 14b/v4 的球速闭环提升：平均 reward `0.4028`、EMA 球速 MAE `0.179 m/s`、球位误差 `0.281`，两轮完整 sequence 的 reward 为 `0.40141/0.40131`。但 v4 的14个手臂 Gaussian raw mean 全部漂到约 `±47--54`，`|tanh(z)|>=0.94` 达到 `99.89%`，执行 residual 几乎恒为 `±0.25 rad`；同时出现5帧右小腿 undesired contact，最大冲击 `395.8 N`。因此 v4 不能替换 Essay 13 baseline。

v5 保留无 PCA 的简单 reference-relative residual，不恢复旧三层 action 处理，只补齐两个互不重复的约束：

1. actor 先输出 raw mean `mu_raw`；14个手臂维度的 Gaussian mean 使用 `mu=atanh(0.95)*tanh(mu_raw/atanh(0.95))` 平滑限制在 `±1.8318`，下肢 mean 不变；这是概率分布参数的数值护栏，不是第二次物理 residual clamp；
2. PPO 仍存储并计算 Gaussian pre-squash sample `z`，不从 float32 的 `±1` action 做 `atanh` 反演；S2 action term 对手臂只执行一次 `a_arm=tanh(z_arm)`；
3. `delta_cmd=0.25*a_arm`，`q_raw=q_ref+delta_cmd`，然后只经过 simulator soft joint limits 和已有 PD actuator；PCA tangent、latent/orthogonal clip、二次 residual envelope 和 1.8 Hz 低通保持删除；
4. 新增 `upper_body_reference_residual_l2`，raw 为14维 executed residual 除以 `0.25 rad` 后的均方，weight `-0.5`。全维饱和时最多约 `-0.010/step`，零 residual 精确为零；它约束长期偏离，`action_rate_l2` 继续只约束帧间变化；
5. S1/S2 的 `actions` observation 仍是最终执行的 normalized absolute joint target，Actor 163 / Critic 295 / Action 29 接口不变；S1、ball reward、contact、termination、reset 和 command 采样均未修改；
6. scalar exploration std 构造 Normal 前仍投影到 `0.05` 下限，NaN/Inf 立即报出 action indices；
7. deterministic play 保存 raw mean、bounded pre-squash mean、post-tanh action 和实际 residual。独立 ONNX 输出 post-tanh arm action，并通过 metadata 明确禁止下游再次 tanh。

v2/v3/v4 checkpoint 切换到 v5 时都不加 migration flag。loader 保留下肢输出、shared/RNN、critic 和 normalizer，只重置14行手臂输出与对应 std，丢弃旧 optimizer，并写入 `regularized_reference_residual_v5`；之后 v5 checkpoint 才正常恢复完整 optimizer。

## 0.4 Essay 14/14b：直接 residual 简化及 v4 饱和

Essay 14 删除 PCA tangent、orthogonal residual 限制、二次 `±0.25 rad` envelope 和 1.8 Hz filter，把 S2 上肢改为14维直接 reference residual。Essay 14b 为避免旧 tanh-Normal 在 float32 饱和 action 上做 `atanh` 反演，将 PPO storage 改为无界 Gaussian pre-squash 变量、由环境执行一次 tanh，并增加 scalar std 下限。

该结构消除了旧执行链的重复约束，但 v4 没有 Gaussian mean 上界，也没有 residual magnitude 正则。由于 observation 和 action-rate 都只看到 tanh 后的执行量，策略可以把 mean 推入数值饱和区，以恒定 `±0.25 rad` residual 隐藏 exploration noise。v5 保留 Essay 14 的简单物理执行链，只修复这个新 null space。

## 0.3 Essay 13：上一版冻结的 full-method baseline

上一版论文主方法冻结在 commit `a589bd71168bd876fe4db93a3d887039c94005a8`（`essay 13`）。`e13.npz` 是该实现同一套 1800-step、两轮手动 command sequence 的代表性诊断，继续作为当前简化候选的直接对照。

| 指标 | E12 | E13 | 当前判断 |
|---|---:|---:|---|
| 平均 `step_reward` | 0.3586 | **0.3721** | 提升 3.8% |
| EMA command-frame 前向球速 | 1.098 m/s | **1.258 m/s** | 速度校准有效 |
| EMA 球速 MAE | 0.357 m/s | **0.219 m/s** | 达到 `<=0.25 m/s` 验收线 |
| 预测球位误差 | 0.369 | **0.301** | 改善 18.5% |
| heading MAE | 0.0609 rad | **0.0565 rad** | 保持稳定 |
| useful-touch 事件数 | 27 | **35** | 触球更积极 |
| 右脚接触间隔中位数 | 1.17 s | **0.86 s** | 典型节奏已改善 |
| undesired-contact 帧数 | 5 | **1** | 接触安全改善 |
| 最大逐-link 接触力 | 638.8 N | **39.7 N** | E12 的小腿异常冲击消失 |
| 实际失败 termination | 0 | **0** | 两轮均完成 |

E13 当时已满足冻结并开始 ablation 的条件：速度、位置、heading、接触安全和 reset 前后平均表现同时稳定，reference 总贡献没有因任务提升而明显下降。当前上肢 action 简化正是一次显式的后续 MDP 修正，因此已触发“建立新 full-method baseline”的要求，不能继续沿用 Essay 13 的名称或结果。

当前已知但不阻塞冻结的边界情况：

- 稳定 `1.5 m/s` command 段的瞬时/EMA 前向球速约为 `1.307/1.281 m/s`，仍低于优选工作区 `1.35--1.60 m/s`，但相对 E12 的 `1.150/1.125 m/s` 已明显提高；
- 典型接触间隔已缩短，但第一次 rollout 在早期触球后出现一次 `2.60 s` 无接触长尾。原因是 startup grace 由绝对 episode age 控制，早期触球不会取消尚未用完的 grace，之后附近可控球又按 `0.5/step` 慢速累计；
- bounded policy 的实际 squashed action P95 为 `0.933`，`|a|>=0.90/0.94` 比例为 `9.50%/3.76%`；内部 boundary pressure 上升，但 projected residual 达到 `0.25 rad` 的比例仅 `0.028%`，executed residual 最大 `0.235 rad`，物理 clamp 没有复发。

## 0.2 Essay 13：速度工作点校准

Essay 12 在 effective command 已达到 `1.5 m/s` 的帧中，瞬时/EMA 前向球速仅约 `1.150/1.126 m/s`。旧 velocity reward 在 `1.325--1.675 m/s` 全部满分，而瞬时安全 penalty 从 `1.70 m/s` 开始；它既给了欠速一个偏宽的平台，又会压制触球后的必要速度峰值。

本次只校准速度工作点，不调整任何 reward 权重或 reference 权重：

- S2 训练 speed 均匀采样从 `0.40--1.50` 扩到 `0.40--1.65 m/s`，使常用 `1.5 m/s` 从分布边界变成内部工作点；
- `dribbling_ball_velocity_tracking` 保持 `+7.5`，改为非对称满分带：欠速容差 `0.05 m/s`、超速容差 `0.20 m/s`；`1.5` command 对应满分区间 `1.45--1.70 m/s`；
- `dribbling_ball_speed_excess` 保持 `-2.5`，`speed_margin` 从 `+0.20` 增为 `+0.30 m/s`；`1.5` command 的瞬时 Huber safety cap 从 `1.70` 移到 `1.80 m/s`，超过 cap 后仍保持非饱和减速梯度；
- diagnostics 新增欠速/超速 excess，并直接保存上肢 `actor_raw_mean`、bounded pre-squash mean 与 squashed action，不再依赖接近 `0.95` 时数值敏感的反推。

## 0.1 Essay 12：contact 语义校验与 bounded policy

`diagnostic_20260819_142724.npz` 暴露出两个不能继续靠后处理掩盖的问题：接触诊断把挂在 `right_knee_link` 下的整段小腿碰撞圆柱显示成“膝盖”，而 S2 actor 的 14 维手臂原始 residual 已有 `89.65%` 落入旧 tanh/clamp 饱和区。

当前修复如下：

- 接触 reward 的合法性不变：只有 `right_ankle_roll_link` 合法，小腿触球仍是 undesired contact；但 diagnostics 同时保存 body 名和碰撞语义名，`right_knee_link` 明确显示为 `right_shin_collision`；
- 逐-link `force_matrix_w` 只有在 runtime filter 列数与配置 filter 数完全一致时才会被赋予 body 名；不再用 `min(count)` 静默截断后继续解释错位列。matrix 存在但数量不符时直接报错，只有旧 runtime 完全没有 matrix 时才走兼容回退；
- reset 后第一个 control step 的 PhysX stale contact sample 统一清零，启动时一次性的左右小腿 `851/877 N` 不再进入 reward、termination 或 diagnostics；
- 修复 contact hold 缓存字段校验错误；两步 hold 现在确实跨控制步保留，并在 reset guard 内清空，不再每次调用时重建；
- S2 手臂 14 维改为部分 tanh-Normal 策略：下肢仍为普通 Gaussian，手臂采样值严格位于 `(-1,1)`，PPO log-prob 包含 tanh Jacobian 修正；
- actor 的手臂 pre-squash mean 先限制在 `±atanh(0.95)`，确定性 play 和 ONNX 使用同一变换；环境收到的已经是有界 residual，不再对它重复施加第一层 tanh；
- PCA 投影后只保留一次 `±0.25 rad` 物理安全 clamp 和 1.8 Hz residual filter，最终仍受 soft joint limits 保护；
- checkpoint action marker 升级为 `bounded_reference_residual_v2`。Essay 11 的 `reference_residual_v1` checkpoint 必须显式做一次 bounded-policy 迁移并重置已经饱和的 14 行手臂输出头。

## 0. Essay 10：reset 与 no-contact 修复

`diagnostic_20260818_131135.npz` 中第一轮在 `+0.65 rad` 和 `-0.65 rad` 转向时分别触发 `dribbling_no_contact`。旧 reset 保留失败时的手动 segment，却沿 task `+X` 重生球，导致球在当前 command frame 中瞬间侧偏 `-0.267 m` / `+0.230 m`；完整 sequence 结束后回到 heading 0，第二轮才明显变好。首帧还出现过 `876.8 N` 的左膝--球初始化接触。

本次修改 reset、no-contact 和 reference action bound，并对 `useful_touch` / `rapid_retouch` 做局部事件尺度修正；不开展大规模 reward 平衡：

- 普通失败 reset 保留当前手动 segment、目标 heading 和剩余 duration，避免后续段永远走不到；
- 只有完整 command sequence 使用 `reset_on_end` 结束时才回到 segment 0；
- resampled command 和 manual segment 都先更新 command，再生成球；
- robot yaw 与球都沿当前有效 command heading 重置；球以实际 randomized reset pelvis 为原点生成；
- S2 初始前向距离收窄为 `0.53 ± 0.05 m`，侧向 jitter 收窄为 `±0.08 m`，降低腿内穿透概率；
- play 在安装 CLI manual command 后立即同步 robot yaw 和球，修复第一次 rollout 仍使用 constructor command frame 的问题；
- `no_contact` 只在 DRIBBLE 生效。普通失控每步 `+1`，距离 `<=0.75 m` 且相对 XY 速度 `<=0.9 m/s` 时每步 `+0.5`，command-change closing recovery 每步 `+0.25`；所有增量都为正，任何有效右 ankle 接触直接清零；
- `useful_touch` 对右脚 `<=14 N` 的新轻触立即给 25% 基础分，剩余 75% 根据 5 steps 后的控制误差改善给出；单次触球 raw score 总上限仍为 1.0。
- `useful_touch` 与 `rapid_retouch` 都按 `score/control_dt` 返回，使配置权重表示每次事件的实际回报；完整有效触球最高 `+1.0/event`，14 steps 内复触为 `-1.0/event`。
- S2 双臂 14 维 action 改为 reference-relative residual：`delta_raw=0.25*a_arm`，先在 motion-bank PCA tangent 中投影修正量，再以 `delta_bound=0.25*tanh(delta_projected/0.25)` 平滑限幅；`a_arm=0` 精确对应当前 motion reference。
- 1.8 Hz 低通只过滤 bounded residual，不再过滤绝对 target；最终执行 `q_exec=clamp(q_ref+delta_filtered, joint_soft_limits)`，reference 换帧不会因旧 target 滞后而越过包络。
- 删除只看旧 clamp 前溢出的 `upper_body_reference_overflow` reward，保留并扩充 residual 饱和度 diagnostics。

网络接口保持 Actor 163 / Critic 295 / Action 29，不需要重训 S1。S2 的 `actions` observation 仍是 action term 真正执行的 normalized joint target，因此 S1/S2 输入含义不变；改变的是 S2 actor 双臂输出的控制语义。旧 checkpoint 必须做一次显式迁移，不能直接 normal resume，具体见 7.1。

## 1. 闭环重构的历史动机

Essay 08 已经把球速压到安全范围，但 `diagnostic_20260818_003056.npz` 表明动作并未恢复到 5.0 水平：

| 指标 | 06 | 07 | Essay 08 | 判断 |
|---|---:|---:|---:|---|
| 球沿 command 前向速度 | 2.348 m/s | 1.221 m/s | 1.346 m/s | 08 的球速合理 |
| 瞬时球速 MAE | 1.172 m/s | 0.425 m/s | 0.402 m/s | 速度闭环有效 |
| `> 2.5 m/s` 比例 | 44.7% | 0.1% | 0.0% | 高速失控已消失 |
| heading error | 0.088 rad | 0.163 rad | 0.155 rad | 转向仍弱于 06 |
| 右脚接触上升沿 | 93 | 64 | 42 | 触球越来越稀疏 |
| 接触间隔中位数 | 18 steps | 20 steps | 37 steps | 08 经常不继续触球 |
| 实际失败 termination | 2 | 4 | 4 | 鲁棒性未恢复 |

失败前的 reward 反算暴露出根因：旧 `dribbling_chase_ball` 只检查 pelvis 的绝对前向速度，超过约 `0.75 m/s` 就满分；即使球正在远离、pelvis 没有真正缩短距离，它仍给满分。与此同时，`orbiting_penalty`、`dribbling_gait_foot_tracking` 和普通 command 速度跟踪会把策略拉回正常步态。结果就是“看起来还在跑，但相对球没有追上”，并在几步后主动减速。

加入球速度观测后，原先用于补偿 position-only 部分可观测性的多个 reward 已不再合适。本次不新增一组 chase reward，而是把正常控球和追球统一为一个物理位置--速度闭环，并把冲突项从代码中直接删除。

## 2. 统一的正常控球/追球闭环

### 2.1 共享状态

在 command 坐标系中定义：

```text
p_rel  = ball_position - pelvis_position
v_rel  = filtered_ball_velocity - pelvis_velocity
p_pred = p_rel + 0.20 * v_rel

p_target = [0.45, 0.00] m
```

球速使用 reset-safe 10-step EMA，控制频率为 50 Hz，因此时间常数约为 0.2 s。`p_pred` 不只看球现在在哪里，还判断按当前相对速度继续运动 0.2 s 后是否会离开可控区。

正常可控区以 `p_target` 为中心：前向半宽 `0.16 m`，侧向半宽 `0.12 m`。超出边界后，`recovery_gate` 在额外 `0.16 m` 范围内从 0 平滑升到 1，并经过 4-step 低通滤波，避免在边界处频繁切换。

### 2.2 正常控球阶段

当 `recovery_gate ≈ 0` 时：

- pelvis 速度目标就是 locomotion command；
- `dribbling_dynamic_proximity` 平滑跟踪预测球位 `[0.45, 0]`；
- `dribbling_ball_velocity_tracking` 让 EMA 球速跟踪 `[command_speed, 0]`；
- 右脚轻触承担离散误差修正；新轻触本身获得一个较小的即时基础分；
- 期望循环是“右脚轻触 → 球自由滚动 → 人跟随 → 误差增大后再次轻触”。

“控球”不等于接触为真，也不等于 30-step possession timer 为真；它表示当前球的位置和速度仍允许下一次正常步态触球完成修正。

### 2.3 追球/恢复阶段

恢复速度目标为：

```text
position_correction = clip_norm(1.5 * (p_pred - p_target), 0.45 m/s)
v_recovery          = clip_norm(v_ball_ema + position_correction, 2.20 m/s)
v_pelvis_target     = (1 - recovery_gate) * v_command
                      + recovery_gate * v_recovery
```

这个公式直接处理三种情况：

- 球太远：correction 指向球，pelvis 比球多出最多 `0.45 m/s` 的追赶速度；
- 球偏侧：correction 同时产生侧向接近速度，不需要 orbiting penalty；
- 球太近：correction 反向，pelvis 自动减速或释放空间；`too_close` 仍作为独立安全屏障。

恢复时球还没被再次触碰，策略无法立刻改变它的自由滚动速度，因此球速正奖励按 `1 - 0.9 × recovery_gate` 降低，最低保留 10%。这让策略先恢复可控几何关系，同时保留回到 command 球速的方向。

### 2.4 有效触球

`dribbling_legal_foot_touch` 改写为 `dribbling_useful_foot_touch`：

1. 检测右 ankle、`<=14 N` 的新接触上升沿；
2. 立即给 `0.25` raw 基础分，承认不改变球速的温和轻触也是合法动作；
3. 同时保存触球前一控制步的综合位置--速度误差；
4. 等待 5 steps（0.1 s）避开碰撞尖峰，再按误差改善给最多 `0.75` raw 附加分；误差下降 `0.50` 时附加分达到上限；
5. 基础分与改善分合计不超过 `1.0` event score；
6. 函数返回 `event_score / control_dt`，抵消 reward manager 的 `×control_dt`，因此一次满分触球的实际回报就是配置权重 `+1.0`；
7. 位于 CG 标注接触窗口内时系数为 1，窗口外为 0.35。CG 只保留为软风格约束，不再独立给 reward。

快速重复触球阈值由 26 steps 降到 14 steps，force 条件同步为 `<=14 N`。06 的成功节奏中位数是 18 steps，因此正常的 18--22 step 触球不再被惩罚，14 steps 内的高频碎点仍受抑制。

## 3. Input 接口

S1 与 S2 的 Actor、Critic 输入顺序和维度完全一致。S1 actor 输出 29 维，S2 actor 输出 23 维；两者反馈给 recurrent observation 的仍是完整 29 维最终关节 target。

### 3.1 Actor：163 维

| 顺序 | Input term | 维度 | 内容 |
|---:|---|---:|---|
| 1 | `command` | 58 | motion reference 的 29 维 joint position + 29 维 joint velocity |
| 2 | `projected_gravity` | 3 | base 坐标系重力方向 |
| 3 | `motion_ref_ang_vel` | 3 | reference anchor angular velocity |
| 4 | `base_ang_vel` | 3 | 机器人 base angular velocity |
| 5 | `joint_pos` | 29 | 相对默认位姿的关节位置 |
| 6 | `joint_vel` | 29 | 关节速度 |
| 7 | `actions` | 29 | 上一帧真正执行的 normalized joint target；S2 肩肘是 `q_ref + 0.25*tanh(z)`、手腕是 exact reference，经 soft joint limit 后共同还原为完整绝对 target |
| 8 | `anchor_ball_polar` | 3 | pelvis 到球的 `[distance, cos(heading), sin(heading)]` |
| 9 | `motion_locomotion_polar_cmd` | 3 | `[speed, cos(command_heading), sin(command_heading)]` |
| 10 | `anchor_ball_velocity_polar_cmd` | 3 | `[ball_xy_speed, cos(delta_heading), sin(delta_heading)]`；静止球为 `[0,0,0]` |

### 3.2 Critic：295 维

| 顺序 | Input term | 维度 | 内容 |
|---:|---|---:|---|
| 1 | `command` | 58 | reference joint position + velocity |
| 2 | `motion_anchor_pos_b` | 3 | reference anchor 在 robot base frame 的位置 |
| 3 | `motion_anchor_ori_b` | 6 | reference anchor 姿态的 6D rotation representation |
| 4 | `body_pos` | 42 | 14 个 tracked body 的 base-frame 位置 |
| 5 | `body_ori` | 84 | 14 个 tracked body 的 6D 姿态 |
| 6 | `base_lin_vel` | 3 | base linear velocity |
| 7 | `base_ang_vel` | 3 | base angular velocity |
| 8 | `joint_pos` | 29 | 关节位置 |
| 9 | `joint_vel` | 29 | 关节速度 |
| 10 | `actions` | 29 | 上一帧 effective action |
| 11 | `anchor_ball_polar` | 3 | 与 Actor 相同的极坐标球位置 |
| 12 | `motion_locomotion_polar_cmd` | 3 | 与 Actor 相同的极坐标 command |
| 13 | `anchor_ball_velocity_polar_cmd` | 3 | 与 Actor 相同的极坐标球速度 |

已删除的笛卡尔球位置、`target_destination_pos_local`、`motion_anchor_lin_vel_cmd` 和 `motion_anchor_ang_vel_cmd` 不再进入网络。Actor 没有独立的 IDLE/DRIBBLE/STOP one-hot；零速状态仍是 counterfactual 输入，而不是单独训练的高层状态。

## 4. 当前完整 Reward 表

控制周期为 `control_dt = 0.005 × 4 = 0.02 s`。表中“每步系数”是 `weight × 0.02`；真实贡献还要乘 reward raw value。Diagnostic 中的 `reward_term_weights` 和 `reward_term_step_weights` 是 runtime 最终值。Stage 2 表的“E13 实际均值”来自 `e13.npz` 的 `reward_terms × 0.02`，表示该固定评测轨迹中 PPO 每步实际收到的分项贡献，不是新的配置权重。

`dribbling_useful_foot_touch` 和 `dribbling_rapid_retouch_penalty` 是离散事件项，raw value 已除以 `control_dt`，所以一次事件的积分回报直接等于 `weight × event_score`，不应再把表中的每步系数当成单次事件强度。

### 4.1 Stage 1 mimic：11 项

任务：`Tracking-CG-G1-Motion-RNN-mimic`

| # | Reward | Weight | 每步系数 | 当前作用 |
|---:|---|---:|---:|---|
| 1 | `motion_body_pos` | +1.0 | +0.020 | pelvis、腿和 torso 的相对位置 mimic |
| 2 | `motion_body_ori` | +1.0 | +0.020 | locomotion body 的相对姿态 mimic |
| 3 | `motion_upper_body_pos` | +0.85 | +0.017 | 双臂关键 link 相对位置 mimic |
| 4 | `motion_upper_body_ori` | +0.35 | +0.007 | 较弱的双臂姿态 mimic |
| 5 | `motion_leg_lin_vel` | +0.4 | +0.008 | 双腿关键 link 线速度跟踪 |
| 6 | `motion_leg_ang_vel` | +0.4 | +0.008 | 双腿关键 link 角速度跟踪 |
| 7 | `motion_anchor_lin_vel` | +2.2 | +0.044 | demo pelvis/root 线速度跟踪 |
| 8 | `motion_anchor_pos_z` | +0.6 | +0.012 | demo anchor 高度跟踪 |
| 9 | `motion_foot_pos` | +0.7 | +0.014 | 左右 ankle-roll 相对位置跟踪 |
| 10 | `action_rate_l2` | -0.1 | -0.002 | 抑制相邻 action 抖动 |
| 11 | `joint_limit` | -10.0 | -0.200 | soft joint limit 越界惩罚 |

S1 没有球任务 reward。球位置、球速度和 locomotion command 仍作为共享输入存在，以保证 S1 checkpoint 可以直接初始化 S2。

### 4.2 Stage 2 control：21 项

任务：`Tracking-CG-G1-Dribbling-RNN-control`

| # | Reward | Weight | 每步系数 | E13 实际均值 | 当前作用与门控 |
|---:|---|---:|---:|---:|---|
| 1 | `motion_body_pos` | +0.45 | +0.009 | +0.007496 | 12 个主要 body 的相对位置风格跟踪；orientation term 已禁用 |
| 2 | `motion_foot_pos` | +0.55 | +0.011 | +0.005964 | 双脚 demo 相对位置跟踪，唯一常驻 gait/foot style 项 |
| 3 | `foot_distance` | +0.35 | +0.007 | +0.006854 | 保持双脚安全间距，阈值 `0.24 m` |
| 4 | `motion_anchor_lin_vel` | +5.0 | +0.100 | +0.079880 | 闭环 pelvis 速度：正常时跟 command，恢复时跟 blended ball-relative target |
| 5 | `motion_anchor_ang_vel` | +1.0 | +0.020 | +0.013007 | heading error 生成 yaw-rate target；gain 2.0，限幅 `1.2 rad/s` |
| 6 | `locomotion_heading_tracking` | +1.0 | +0.020 | +0.019537 | command heading 跟踪；`0--0.25 m/s` 速度门控 |
| 7 | `dribbling_dynamic_proximity` | +3.0 | +0.060 | +0.053644 | 平滑跟踪 0.2 s 预测球位 `[0.45,0]`，不依赖 contact timer |
| 8 | `dribbling_ball_too_close_penalty` | -8.0 | -0.160 | 0.000000 | pelvis--ball XY 距离 `<0.28 m` 起罚，`<=0.14 m` 满罚 |
| 9 | `dribbling_pelvis_quat_tracking` | +2.0 | +0.040 | +0.036251 | pelvis 姿态跟踪 motion reference，`std=0.45` |
| 10 | `dribbling_ball_velocity_tracking` | +7.5 | +0.150 | +0.091167 | 10-step EMA 球速非对称双向跟踪；欠速/超速容差 `0.05/0.20 m/s`，由 controllability gate 在恢复期降权 |
| 11 | `dribbling_ball_speed_excess` | -2.5 | -0.050 | -0.000308 | 瞬时 XY 球速 command-relative Huber 安全项，cap=`max(0.35,target+0.30)` |
| 12 | `dribbling_sustained_contact_penalty` | -6.0 | -0.120 | -0.000014 | 20-step per-link contact duty EMA；25% 起罚、60% 满罚 |
| 13 | `dribbling_ball_bounce_penalty` | -3.0 | -0.060 | -0.000367 | 接触时球垂直速度绝对值超过 `0.32 m/s` 起罚 |
| 14 | `dribbling_rapid_retouch_penalty` | -1.0 | -0.020 | -0.001111 | event-normalized；两次右脚 `<=14 N` 触球间隔 `<14 steps` 时实际回报 `-1.0/event` |
| 15 | `dribbling_useful_foot_touch` | +1.0 | +0.020 | +0.013387 | event-normalized；新轻触给 25% 基础分，5 steps 后按控制改善给其余 75%，单次最高 `+1.0`；CG 窗口为软系数 |
| 16 | `dribbling_micro_contact_filter` | -4.0 | -0.080 | 0.000000 | 右 ankle force EMA 超过 `22 N` 后连续惩罚，raw 上限 2 |
| 17 | `dribbling_undesired_contact_penalty` | -12.0 | -0.240 | -0.000133 | 右 ankle 以外所有机器人碰撞体触球均惩罚，包括左脚和左右小腿；URDF 的小腿碰撞体挂在 `*_knee_link` 下 |
| 18 | `dribbling_cg_foot_ball_distance` | +3.5 | +0.070 | +0.062237 | 仅在 CG reference foot--ball distance `<=0.55 m` 的接近/触球窗口跟踪，`std=0.22` |
| 19 | `action_rate_l2` | -0.1 | -0.002 | -0.011339 | 当前代码：下肢对 effective action、手臂对实际 reference residual 做变化率惩罚；E13 数值来自旧 absolute-effective 定义，仅作历史对照 |
| 20 | `upper_body_pre_squash_action_l2` | -0.15 | -0.003 | N/A | v6：8维肩肘 Gaussian pre-squash `z` 的均方；不含6个 reference-only wrist。`|z|=1.8318` 时约 `-0.0101/step`，继续增大时不饱和 |
| 21 | `joint_limit` | -10.0 | -0.200 | -0.004023 | soft joint limit 越界惩罚 |

球速正预算仍为 `+7.5`，没有比 5.0/06 扩张。区别是 Essay 08 的 `progress +5.0` 和 `velocity +2.5` 已合并为一个 `velocity +7.5`：它同时惩罚过慢、过快和侧向速度，不再存在只要求“至少够快”的单边目标。

E13 的 runtime 分组均值为：reference（`body_pos + foot_pos + pelvis_quat + cg_foot_ball_distance`）`+0.111948/step`；带球连续项与事件正奖励 `+0.270622/step`，带球安全惩罚 `-0.001933/step`，带球净贡献 `+0.268689/step`；通用 `foot_distance + action_rate + joint_limit` 净贡献 `-0.008508/step`。带球/reference 净比例约 `2.40:1`，总和为 `0.372129/step`。安全惩罚在成功轨迹中应当稀疏，不能仅凭接近零的长期均值判断其早期训练作用为零。

## 5. 当前已删除/合并的 Reward

| Essay 08 reward | 原权重 | 处理 | 删除原因 |
|---|---:|---|---|
| `dribbling_face_ball` | +1.0 | 删除 | heading tracking 已约束 pelvis 朝向；预测球位又约束球在 command 前方，重复 |
| `dribbling_ball_coast_penalty` | -2.2 | 删除并入 velocity tracking | 正确速度的自由滚动本来就是正常控球阶段，不应因暂时无接触受罚 |
| `dribbling_ball_forward_progress` | +5.0 | 删除并入 velocity tracking | 只有速度下界，会鼓励过快球；现直接跟踪完整速度向量 |
| `dribbling_orbiting_penalty` | -6.0 | 删除并入 pelvis 闭环 | 旧项在恢复期平均可抵消甚至超过 chase；侧向位置误差现在直接产生接近速度 |
| `dribbling_gait_foot_tracking` | +1.4 | 删除 | 与常驻 `motion_foot_pos` 重复，而且恰在追球时把策略拉回 demo gait |
| `dribbling_chase_ball` | +2.0 | 删除并入 pelvis 闭环 | 只奖励绝对 pelvis 前向速度，不能判断是否真正接近运动中的球 |
| `dribbling_cg_contact_consistency` | +1.0 | 合并进 useful touch | 独立 CG 事件分数不判断触球结果；现在只对物理上有效的触球做 timing 调制 |
| `upper_body_reference_overflow` | -0.05 | 删除 | 读取旧绝对 target 包络前的 overflow，但策略输入是执行后的 action；当前由 action term 中的一次 tanh 与统一 reference residual margin 直接定义执行语义，joint limit 另作物理安全保护 |

这些项不是权重设为 0，而是已从 Stage 2 配置及对应实现中删除。Essay 14 时 S2 reward 数量由 28 降为 20；v5 新增的 post-tanh residual 正则在 v6 被 pre-squash 正则一对一替换，因此当前仍为21项。

## 6. Termination

### 6.1 Stage 1 mimic

| Termination | 条件 |
|---|---|
| `time_out` | 10 s，即 500 control steps |
| `anchor_pos_z` | reference 与 robot anchor 高度误差超过 `0.32 m` |
| `anchor_ori` | anchor orientation error 超过 `0.8 rad` |
| `ee_body_pos` | 20-step reset grace 后，左右脚或左右手的 Z 误差超过 `0.25 m` |

### 6.2 Stage 2 control

| Termination | 条件 |
|---|---|
| `time_out` | 10 s，即 500 control steps |
| `ball_lost` | 仅 DRIBBLE：50-step grace 后，pelvis--ball XY 距离 `>1.0 m` 或瞬时 XY 速度差 `>2.0 m/s` |
| `dribbling_no_contact` | 仅 DRIBBLE：50-step grace 后，任何有效右 ankle 接触直接清零。普通状态 `+1/step`（1 s 到阈值），近球且相对速度可控时 `+0.5/step`（2 s），command-change 且正在 closing 时 `+0.25/step`（4 s）；累计到 50 结束，永不冻结或倒减 |
| `locomotion_manual_sequence_end` | play 使用 `--locomotion_cmd_reset_on_end` 且手动 command sequence 完成 |

除删除与新 smooth bound 冲突的 `upper_body_reference_overflow`，以及对 useful/rapid touch 的局部事件尺度修正外，本次没有整体重排 reward。`no_contact` 保持安全终止职责；正常触球节奏由有意义的正负事件回报学习，而不是继续收紧 termination 强制实现。

当前冻结实现中的 `grace_steps=50` 只检查绝对 `episode_length_buf`，不是“直到首次触球”的状态机：如果策略在第 50 step 前已经触球，剩余 startup grace 仍继续生效。E13 第一轮在 step 12 触球后出现 `2.60 s` 无接触长尾正是这个边界情况；附近可控球的慢速计数最高只到 `42.75/50`。Essay 13 ablation 保持此逻辑不变并对所有 variant 使用同一 termination；报告接触间隔时同时给出 median、P90 和包含 reset 过渡的 max，不隐藏该长尾。若以后改为 `ever_contact` 锁存并提前结束 grace，该结果属于新 baseline。

## 7. Reset 与 command 行为

- control step 为 0.02 s；默认 episode 10 s。
- reset 时 robot 和 ball 一起重置，robot/ball 初始速度清零；球从实际 randomized pelvis 出发，沿 active command heading 生成，前向距离为 `0.48--0.58 m`、侧向 jitter 为 `±0.08 m`。
- S2 的 motion clip 作为循环 style phase 使用，clip 结束不会单独重置场景。
- play 中任意 episode termination 都会清空 RNN hidden state；普通失败保留当前手动 segment、heading 和剩余 duration，重置后的 robot yaw 与球使用当前有效 heading。只有完整 sequence 的 `reset_on_end` 会回到 segment 0。
- CLI manual command 安装后会立即同步 robot yaw 和球，因此第一次 rollout 与之后的 reset 使用相同 command frame。
- command 训练范围为 speed `0.40--1.65 m/s`、heading `-0.75--0.75 rad`、duration `3--6 s`；`1.5 m/s` 不再位于训练上边界。
- 输入维度保持 Actor 163、Critic 295，反馈 observation 中的 `actions` 保持29维。S1 policy action 仍为29维，S2 policy action 改为23维；不需要重训 S1，但旧 S2 输出头需要一次性迁移。
- reset 后首个 control step 的接触传感器值按初始化保护帧处理，不会把 pre-reset/stale PhysX 冲量计入触球 reward、termination 或 diagnostics。
- Essay 13 两轮评测的平均 reward 与 EMA 球速分别为 `0.37235/0.37191` 和 `1.259/1.257 m/s`，说明第一次与第二次 rollout 的平均控制表现已经一致；第一轮的单次长接触间隔属于上述 grace 长尾，不是旧 command-frame 球位置错位复发。

### 7.1 旧 checkpoint 的一次性迁移

保留 checkpoint 分为三种 action 语义，迁移参数不能混用：

1. 旧 S1/S2 checkpoint 没有 action marker，双臂还是绝对 target。首次初始化 S2 时添加：

```bash
--migrate_legacy_upper_body_residual
```

标准 `shell/progressive_dribbling_train.sh` 已在 S1→S2 边界自动添加该参数；用户不需要手工追加。它只作用于这一次加载，S2 后续 resume 不得继续携带。

2. Essay 11 的 S2 checkpoint 带 `reference_residual_v1` marker，但其 Gaussian actor 仍可输出无界 residual。首次切换到当前 bounded policy 时添加：

```bash
--migrate_bounded_upper_body_policy
```

3. Essay 12/13 checkpoint 带 `bounded_reference_residual_v2`，失败尝试可能带 `direct_reference_residual_v3`，Essay 14b 带 `pre_squash_reference_residual_v4`，Essay 15 带 `regularized_reference_residual_v5`。V2--v5 切换到当前 v6 都不加 migration 参数；loader 保留 shared/RNN、critic、normalizer 和15个下肢/腰部输出行，删除6个 wrist 行，重置8个肩肘输出行及对应 std，并丢弃旧 optimizer。

4. 当前版本保存的 checkpoint 带 `shoulder_elbow_reference_residual_v6` marker，23维输出结构一致时正常 resume 并恢复 optimizer；两个 migration 参数都不能添加。

所有迁移路径都把旧29维输出重排为当前23维：旧输出中与15个下肢/腰部物理关节对应的行原样复制，6个 wrist 输出被移除，8个肩肘输出清零且 exploration std 设为 `0.5`；RNN/MLP 隐层、critic 和 observation normalizer 全部保留。由于输出张量和语义变化，旧 optimizer 被丢弃并重新初始化。

这是带 `TEMPORARY LEGACY MIGRATION` 标记的过渡模块。无 marker/v1 需要显式参数，v2--v5 到 v6 则按 marker 自动触发一次：

- 无 marker 或 `reference_residual_v1` checkpoint 未提供各自正确的迁移参数时，loader 直接拒绝加载；
- 当前新 checkpoint 会在 `infos` 中写入 `upper_body_action_semantics=shoulder_elbow_reference_residual_v6`；
- 正确 resume 新 checkpoint 时**不得**再加迁移参数；如果误加，loader 会报错而不是再次清零双臂；
- 基线确认且所有保留 checkpoint 都已迁移后，可以删除这两个 CLI 参数和 checkpoint loader 中整段临时迁移代码；runtime bounded policy 与 residual action 不依赖它。

## 8. Diagnostics

`--diagnostic` 继续保存实际 runtime reward 名称、权重、每步权重和逐项贡献：

- `reward_term_names`
- `reward_term_weights`
- `reward_term_step_weights`
- `reward_step_dt`
- `reward_terms`
- `step_reward`

当前保存的闭环与 action-quality 字段：

- `ball_predicted_forward_offset`、`ball_predicted_lateral_offset`
- `ball_position_error_norm`、`ball_control_position_reward`
- `recovery_gate_raw`、`recovery_gate`
- `ball_velocity_controllability_gate`
- `ball_closing_speed`
- `pelvis_command_target_velocity_xy`
- `pelvis_position_correction_velocity_xy`
- `pelvis_recovery_target_velocity_xy`
- `pelvis_blended_target_velocity_xy`
- `closed_loop_pelvis_velocity_reward`
- `no_contact_count`、`no_contact_increment`
- `useful_touch_new_event`、`useful_touch_evaluated`、`useful_touch_pending`
- `useful_touch_pre_error`、`useful_touch_current_error`、`useful_touch_improvement`
- `useful_touch_cg_aligned`、`useful_touch_base_reward`
- `useful_touch_improvement_reward`、`useful_touch_event_score`
- `useful_touch_reward`、`rapid_retouch_event`、`rapid_retouch_reward`
- `reference_relative_upper_body_residual`
- `upper_reference_target`、`upper_raw_target`、`upper_executed_target`
- `upper_residual_pre_squash`、`upper_residual_policy`、`upper_residual_commanded`、`upper_residual_executed`
- `upper_residual_actor_boundary_fraction`：8个 learned shoulder/elbow 中 `|tanh(z)|>=0.95` 的比例
- `upper_residual_joint_limit_fraction`：最终 soft joint limit 实际截断比例
- `upper_actor_raw_mean`、`upper_actor_squashed_action`：分别保存8个 learned shoulder/elbow 的 Gaussian mean 与 post-tanh action；14维数组中6个 reference-only wrist 的 policy telemetry 为 NaN
- `policy_action_dim`、`policy_action_ids`：记录当前23维 policy 接口；reference-only wrist 对应 `-1`
- `residual_upper_joint_names`、`residual_upper_local_ids`、`reference_only_upper_joint_names`：明确区分8个 learned residual 与6个 exact-reference wrist
- `ball_filtered_underspeed_error`、`ball_filtered_overspeed_error`
- `ball_velocity_underspeed_tolerance`、`ball_velocity_overspeed_tolerance`
- `ball_contact_body_names`：Isaac articulation body/link 名
- `ball_contact_collision_names`：用于视频解释的碰撞语义名；例如 `right_knee_link -> right_shin_collision`
- `ball_contact_filter_count`、`ball_contact_expected_filter_count`
- `ball_contact_filter_mapping_valid`：只有为 true 时逐-link force 列才按上述名称解释
- `bounded_upper_body_policy_action`：当前 action term 是否把 Gaussian pre-squash 手臂变量通过一次 tanh 映射为有界 residual；S2 为 true，S1 为 false

旧 possession timer、独立 CG contact-consistency 与全部 manifold/PCA telemetry 已删除，避免把不再参与当前 MDP 的状态误认为有效门控。

## 9. Essay 13 历史验收结果与新 baseline 约束

### 9.1 冻结时的验收结果

| 指标 | 目标 | E13 | 状态 |
|---|---:|---:|---|
| EMA 球速 MAE | `<=0.25 m/s` | **0.219 m/s** | 通过 |
| 稳定 `1.5 m/s` command 瞬时/EMA 前向球速 | 优选 `1.35--1.60 m/s` | **1.307/1.281 m/s** | 未到优选区，但相对 E12 明显改善；冻结接受 |
| 球速 `>2.5 m/s` 比例 | `<5%` | **0%** | 通过 |
| 平均绝对 heading error | `<=0.12 rad` | **0.0565 rad** | 通过 |
| 1800-step 实际失败 termination | `<=2` | **0** | 通过 |
| 右脚接触间隔中位数 | `0.7--1.0 s` | **0.86 s** | 通过 |
| 右脚接触间隔 P90 | `<=1.5 s` 为优选 | **1.55 s** | 接近；保留监控 |
| 包含 reset 过渡的最长内部无接触段 | 必须报告 | **2.60 s** | 已知 startup-grace 长尾 |
| undesired-contact 帧数 | 越低越好 | **1/1800** | 通过 |
| 最大逐-link 接触力 | 不出现数百牛初始化/小腿异常冲击 | **39.7 N** | 通过 |
| reference 总贡献 | 不因任务提升显著坍缩 | **0.111948/step**；E12 为 0.112546 | 通过 |
| projected residual 达到 `0.25 rad` | `<1%` | **0.028%** | 通过 |
| squashed action `>=0.90/0.94` | 直接报告 boundary pressure | **9.50%/3.76%** | 可接受，继续作为论文指标 |

这些结果说明 Essay 13 已完成“可冻结的论文主方法原型”，但单个 checkpoint、单个 seed 和一条 1800-step diagnostic 不能单独构成论文统计证据。当前上肢 action 简化必须先用同一评测协议与 E13 对照；达到不降低带球成功率、球速、heading、接触安全且显著降低 shoulder-pitch 幅值/速度后，才能把简化版本冻结为新的 full method，再开展其 ablation。

### 9.2 Ablation 公平性约束

- `essay 13` commit `a589bd71168bd876fe4db93a3d887039c94005a8` 及 `e13.npz` 固定为上一版对照；当前简化代码在通过同协议验收前只是候选，不能混用两个实现的结果。
- 所有 variant 使用相同 S1 初始化、S2 motion clip、训练步数、seed 集合、command curriculum、reset/termination 和 evaluation command；每次只移除或替换被考察模块。
- 至少报告 3 个独立训练 seed 的均值和标准差；算力允许时优先 5 个 seed。不能挑选单个最好 checkpoint 代替跨 seed 结果。
- 统一评测至少覆盖 speed `0.5/1.0/1.5 m/s` 与 heading `0/+0.65/-0.65 rad`，并分开报告直行、转向、reset 后第一轮和连续第二轮。
- 核心指标固定为：成功率、EMA 球速 MAE、预测球位误差、heading MAE、接触间隔 median/P90/max、useful/rapid/undesired contact、最大逐-link force、reference 跟踪误差、action-rate/joint-limit 贡献及 upper-body boundary pressure。
- 所有 variant 从 diagnostic 的 `reward_term_weights`、`reward_term_step_weights` 和 `reward_terms` 读取实际 runtime 权重与贡献；不根据配置文件手工推测。
- Ablation 期间不修 no-contact grace、不改 reward 权重，也不清理一次性 checkpoint 迁移模块。任何这些改变都需要新 commit、新 full baseline，并让所有 variant 重新使用同一实现。

当前 S2 输出层从29维变为23维。无 marker 的旧 S1/S2 使用 `--migrate_legacy_upper_body_residual`；Essay 11 `reference_residual_v1` 使用 `--migrate_bounded_upper_body_policy`。Essay 12/13 v2、失败尝试 v3、Essay 14b v4 和 Essay 15 v5 都不加 flag，自动保留 shared/RNN/critic/normalizer 与15个下肢/腰部输出，移除 wrist 输出并初始化8个肩肘 residual。之后从 `shoulder_elbow_reference_residual_v6` checkpoint 正常 resume，绝不能继续携带迁移参数。
