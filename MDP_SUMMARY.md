# Footmimic 当前 MDP 总结：S1 mimic + S2 control

> **维护约定：本文件是当前 S1/S2 MDP 的唯一持续更新总表。** 后续 input、reward、termination、reset 或 diagnostics 发生变化时，直接修改 `MDP_SUMMARY.md`，不再新建 `MDP_SUMMARY_09.md` 等编号文件；旧的编号版中间总结已删除。

## 0.2 当前更新：速度工作点校准

Essay 12 在稳定 `1.5 m/s` command 段的瞬时/EMA 前向球速仅约 `1.172/1.146 m/s`。旧 velocity reward 在 `1.325--1.675 m/s` 全部满分，而瞬时安全 penalty 从 `1.70 m/s` 开始；它既给了欠速一个偏宽的平台，又会压制触球后的必要速度峰值。

本次只校准速度工作点，不调整任何 reward 权重或 reference 权重：

- S2 训练 speed 均匀采样从 `0.40--1.50` 扩到 `0.40--1.65 m/s`，使常用 `1.5 m/s` 从分布边界变成内部工作点；
- `dribbling_ball_velocity_tracking` 保持 `+7.5`，改为非对称满分带：欠速容差 `0.05 m/s`、超速容差 `0.20 m/s`；`1.5` command 对应满分区间 `1.45--1.70 m/s`；
- `dribbling_ball_speed_excess` 保持 `-2.5`，`speed_margin` 从 `+0.20` 增为 `+0.30 m/s`；`1.5` command 的瞬时 Huber safety cap 从 `1.70` 移到 `1.80 m/s`，超过 cap 后仍保持非饱和减速梯度；
- diagnostics 新增欠速/超速 excess，并直接保存上肢 `actor_raw_mean`、bounded pre-squash mean 与 squashed action，不再依赖接近 `0.95` 时数值敏感的反推。

## 0.1 当前更新：contact 语义校验与 bounded policy

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

## 1. 本次更新解决什么问题

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

S1 与 S2 的 Actor、Critic 输入顺序和维度完全一致；本次 reward 重构没有改变 checkpoint 的网络结构。

### 3.1 Actor：163 维

| 顺序 | Input term | 维度 | 内容 |
|---:|---|---:|---|
| 1 | `command` | 58 | motion reference 的 29 维 joint position + 29 维 joint velocity |
| 2 | `projected_gravity` | 3 | base 坐标系重力方向 |
| 3 | `motion_ref_ang_vel` | 3 | reference anchor angular velocity |
| 4 | `base_ang_vel` | 3 | 机器人 base angular velocity |
| 5 | `joint_pos` | 29 | 相对默认位姿的关节位置 |
| 6 | `joint_vel` | 29 | 关节速度 |
| 7 | `actions` | 29 | 上一帧真正执行的 normalized joint action；S2 上身是 smooth reference bound、manifold projection、低通滤波之后的结果 |
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

控制周期为 `control_dt = 0.005 × 4 = 0.02 s`。表中“每步系数”是 `weight × 0.02`；真实贡献还要乘 reward raw value。Diagnostic 中的 `reward_term_weights` 和 `reward_term_step_weights` 是 runtime 最终值。

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

### 4.2 Stage 2 control：20 项

任务：`Tracking-CG-G1-Dribbling-RNN-control`

| # | Reward | Weight | 每步系数 | 当前作用与门控 |
|---:|---|---:|---:|---|
| 1 | `motion_body_pos` | +0.45 | +0.009 | 12 个主要 body 的相对位置风格跟踪；orientation term 已禁用 |
| 2 | `motion_foot_pos` | +0.55 | +0.011 | 双脚 demo 相对位置跟踪，唯一常驻 gait/foot style 项 |
| 3 | `foot_distance` | +0.35 | +0.007 | 保持双脚安全间距，阈值 `0.24 m` |
| 4 | `motion_anchor_lin_vel` | +5.0 | +0.100 | 闭环 pelvis 速度：正常时跟 command，恢复时跟 blended ball-relative target |
| 5 | `motion_anchor_ang_vel` | +1.0 | +0.020 | heading error 生成 yaw-rate target；gain 2.0，限幅 `1.2 rad/s` |
| 6 | `locomotion_heading_tracking` | +1.0 | +0.020 | command heading 跟踪；`0--0.25 m/s` 速度门控 |
| 7 | `dribbling_dynamic_proximity` | +3.0 | +0.060 | 平滑跟踪 0.2 s 预测球位 `[0.45,0]`，不依赖 contact timer |
| 8 | `dribbling_ball_too_close_penalty` | -8.0 | -0.160 | pelvis--ball XY 距离 `<0.28 m` 起罚，`<=0.14 m` 满罚 |
| 9 | `dribbling_pelvis_quat_tracking` | +2.0 | +0.040 | pelvis 姿态跟踪 motion reference，`std=0.45` |
| 10 | `dribbling_ball_velocity_tracking` | +7.5 | +0.150 | 10-step EMA 球速非对称双向跟踪；欠速/超速容差 `0.05/0.20 m/s`，由 controllability gate 在恢复期降权 |
| 11 | `dribbling_ball_speed_excess` | -2.5 | -0.050 | 瞬时 XY 球速 command-relative Huber 安全项，cap=`max(0.35,target+0.30)` |
| 12 | `dribbling_sustained_contact_penalty` | -6.0 | -0.120 | 20-step per-link contact duty EMA；25% 起罚、60% 满罚 |
| 13 | `dribbling_ball_bounce_penalty` | -3.0 | -0.060 | 接触时球垂直速度绝对值超过 `0.32 m/s` 起罚 |
| 14 | `dribbling_rapid_retouch_penalty` | -1.0 | -0.020 | event-normalized；两次右脚 `<=14 N` 触球间隔 `<14 steps` 时实际回报 `-1.0/event` |
| 15 | `dribbling_useful_foot_touch` | +1.0 | +0.020 | event-normalized；新轻触给 25% 基础分，5 steps 后按控制改善给其余 75%，单次最高 `+1.0`；CG 窗口为软系数 |
| 16 | `dribbling_micro_contact_filter` | -4.0 | -0.080 | 右 ankle force EMA 超过 `22 N` 后连续惩罚，raw 上限 2 |
| 17 | `dribbling_undesired_contact_penalty` | -12.0 | -0.240 | 右 ankle 以外所有机器人碰撞体触球均惩罚，包括左脚和左右小腿；URDF 的小腿碰撞体挂在 `*_knee_link` 下 |
| 18 | `dribbling_cg_foot_ball_distance` | +3.5 | +0.070 | 仅在 CG reference foot--ball distance `<=0.55 m` 的接近/触球窗口跟踪，`std=0.22` |
| 19 | `action_rate_l2` | -0.1 | -0.002 | 对 effective action 做变化率惩罚 |
| 20 | `joint_limit` | -10.0 | -0.200 | soft joint limit 越界惩罚 |

球速正预算仍为 `+7.5`，没有比 5.0/06 扩张。区别是 Essay 08 的 `progress +5.0` 和 `velocity +2.5` 已合并为一个 `velocity +7.5`：它同时惩罚过慢、过快和侧向速度，不再存在只要求“至少够快”的单边目标。

## 5. 本次直接删除/合并的 Reward

| Essay 08 reward | 原权重 | 处理 | 删除原因 |
|---|---:|---|---|
| `dribbling_face_ball` | +1.0 | 删除 | heading tracking 已约束 pelvis 朝向；预测球位又约束球在 command 前方，重复 |
| `dribbling_ball_coast_penalty` | -2.2 | 删除并入 velocity tracking | 正确速度的自由滚动本来就是正常控球阶段，不应因暂时无接触受罚 |
| `dribbling_ball_forward_progress` | +5.0 | 删除并入 velocity tracking | 只有速度下界，会鼓励过快球；现直接跟踪完整速度向量 |
| `dribbling_orbiting_penalty` | -6.0 | 删除并入 pelvis 闭环 | 旧项在恢复期平均可抵消甚至超过 chase；侧向位置误差现在直接产生接近速度 |
| `dribbling_gait_foot_tracking` | +1.4 | 删除 | 与常驻 `motion_foot_pos` 重复，而且恰在追球时把策略拉回 demo gait |
| `dribbling_chase_ball` | +2.0 | 删除并入 pelvis 闭环 | 只奖励绝对 pelvis 前向速度，不能判断是否真正接近运动中的球 |
| `dribbling_cg_contact_consistency` | +1.0 | 合并进 useful touch | 独立 CG 事件分数不判断触球结果；现在只对物理上有效的触球做 timing 调制 |
| `upper_body_reference_overflow` | -0.05 | 删除 | 读取旧绝对 target 包络前的 overflow，但策略输入是执行后的 action；现由 reference-relative residual、PCA tangent 投影、策略侧 tanh-Normal 和投影后的物理安全限幅直接定义执行语义 |

这些项不是权重设为 0，而是已从 Stage 2 配置及对应实现中删除。S2 reward 数量由 28 降为 20。

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

## 7. Reset 与 command 行为

- control step 为 0.02 s；默认 episode 10 s。
- reset 时 robot 和 ball 一起重置，robot/ball 初始速度清零；球从实际 randomized pelvis 出发，沿 active command heading 生成，前向距离为 `0.48--0.58 m`、侧向 jitter 为 `±0.08 m`。
- S2 的 motion clip 作为循环 style phase 使用，clip 结束不会单独重置场景。
- play 中任意 episode termination 都会清空 RNN hidden state；普通失败保留当前手动 segment、heading 和剩余 duration，重置后的 robot yaw 与球使用当前有效 heading。只有完整 sequence 的 `reset_on_end` 会回到 segment 0。
- CLI manual command 安装后会立即同步 robot yaw 和球，因此第一次 rollout 与之后的 reset 使用相同 command frame。
- command 训练范围为 speed `0.40--1.65 m/s`、heading `-0.75--0.75 rad`、duration `3--6 s`；`1.5 m/s` 不再位于训练上边界。
- 输入与 action 维度不变：Actor 163、Critic 295、Action 29，因此现有 S1/S2 checkpoint 在结构上兼容；本次不需要重训 S1。
- reset 后首个 control step 的接触传感器值按初始化保护帧处理，不会把 pre-reset/stale PhysX 冲量计入触球 reward、termination 或 diagnostics。

### 7.1 旧 checkpoint 的一次性迁移

保留 checkpoint 分为三种 action 语义，迁移参数不能混用：

1. 旧 S1/S2 checkpoint 没有 action marker，双臂还是绝对 target。首次初始化 S2 时添加：

```bash
--migrate_legacy_upper_body_residual
```

2. Essay 11 的 S2 checkpoint 带 `reference_residual_v1` marker，但其 Gaussian actor 仍可输出无界 residual。首次切换到当前 bounded policy 时添加：

```bash
--migrate_bounded_upper_body_policy
```

3. 当前 checkpoint 带 `bounded_reference_residual_v2` marker，正常 resume，两个迁移参数都不能添加。

两条迁移路径都只清零 actor 最后输出层的双臂 14 行，将这 14 维 exploration std 设为 `0.5`；下肢输出、RNN/MLP 隐层、critic 和 observation normalizer 全部保留。由于 action 分布或语义变化，旧 optimizer 被丢弃并重新初始化。

这是带 `TEMPORARY LEGACY MIGRATION` 标记的过渡模块，不会自动触发：

- 无 marker 或 `reference_residual_v1` checkpoint 未提供各自正确的迁移参数时，loader 直接拒绝加载；
- 迁移后的新 checkpoint 会在 `infos` 中写入 `upper_body_action_semantics=bounded_reference_residual_v2`；
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

本次新增或改写的闭环字段：

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
- `manifold_reference_raw_deviation`
- `manifold_reference_bounded_deviation`
- `manifold_reference_post_projection_deviation`
- `manifold_reference_saturation_fraction`
- `reference_relative_upper_body_residual`
- `upper_residual_policy`、`upper_residual_projected`
- `upper_residual_executed`、`upper_residual_saturation_fraction`
- `upper_actor_raw_mean`、`upper_actor_bounded_mean`、`upper_actor_squashed_action`
- `ball_filtered_underspeed_error`、`ball_filtered_overspeed_error`
- `ball_velocity_underspeed_tolerance`、`ball_velocity_overspeed_tolerance`
- `ball_contact_body_names`：Isaac articulation body/link 名
- `ball_contact_collision_names`：用于视频解释的碰撞语义名；例如 `right_knee_link -> right_shin_collision`
- `ball_contact_filter_count`、`ball_contact_expected_filter_count`
- `ball_contact_filter_mapping_valid`：只有为 true 时逐-link force 列才按上述名称解释
- `bounded_upper_body_policy_action`：当前 action term 是否启用策略侧 tanh-Normal；新 S2 应为 true，S1 为 false

旧 possession timer 和独立 CG contact-consistency telemetry 已删除，避免把不再参与当前 MDP 的状态误认为有效门控。

## 9. 下一次训练的验收重点

| 指标 | 目标 |
|---|---:|
| EMA 球速 MAE | `<=0.25 m/s` |
| `1.5 m/s` command 平均前向球速 | 优先达到 `1.35--1.60 m/s`，同时避免只靠高冲击峰值抬均值 |
| 球速 `>2.5 m/s` 比例 | `<5%` |
| 平均绝对 heading error | `<=0.12 rad` |
| 1800-step 实际失败 termination | `<=2` |
| 右脚接触间隔中位数 | 先恢复到约 `25--40 steps`，同时 `<14 steps` 高频触球受控 |
| recovery 中 closing speed | 大部分为正，不再出现 chase reward 满分但距离持续增大 |
| useful touch 改善成功率 | 随训练上升；轻触基础分与延迟改善分必须分开看 |
| 手臂、腰部参考误差 | 至少保持 06 水平，并继续向 5.0 靠近 |
| 上肢 residual 饱和比例 | `upper_residual_policy` 必须始终位于 `[-1,1]`；同时直接检查 `upper_actor_raw_mean` P95 与 squashed action 的 `>=0.90/0.94` boundary pressure |

网络参数结构不变，但旧 checkpoint 的双臂输出分布或语义不同。无 marker 的旧 S1/S2 使用 `--migrate_legacy_upper_body_residual`；Essay 11 `reference_residual_v1` 使用 `--migrate_bounded_upper_body_policy`。两者都只迁移一次并写入新 run，之后从 `bounded_reference_residual_v2` checkpoint 正常 resume，绝不能继续携带迁移参数。
