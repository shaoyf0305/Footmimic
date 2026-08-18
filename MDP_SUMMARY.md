# Footmimic 当前 MDP 总结：S1 mimic + S2 control

> **维护约定：本文件是当前 S1/S2 MDP 的唯一持续更新总表。** 后续 input、reward、termination、reset 或 diagnostics 发生变化时，直接修改 `MDP_SUMMARY.md`，不再新建 `MDP_SUMMARY_09.md` 等编号文件；旧的编号版中间总结已删除。

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
- 右脚触球只承担离散误差修正，不因“碰到球”本身得分；
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
2. 保存触球前一控制步的综合位置--速度误差；
3. 等待 5 steps（0.1 s），避开碰撞尖峰；
4. 只有误差下降才给正奖励，下降 `0.50` 达到满分；
5. 位于 CG 标注接触窗口内时系数为 1，窗口外为 0.35。CG 只保留为软风格约束，不再独立给 reward。

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
| 7 | `actions` | 29 | 上一帧真正执行的 normalized joint action |
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
| 10 | `dribbling_ball_velocity_tracking` | +7.5 | +0.150 | 10-step EMA 球速双向跟踪；由 controllability gate 在恢复期降权 |
| 11 | `dribbling_ball_speed_excess` | -2.5 | -0.050 | 瞬时 XY 球速 command-relative Huber 安全项，cap=`max(0.35,target+0.20)` |
| 12 | `dribbling_sustained_contact_penalty` | -6.0 | -0.120 | 20-step per-link contact duty EMA；25% 起罚、60% 满罚 |
| 13 | `dribbling_ball_bounce_penalty` | -3.0 | -0.060 | 接触时球垂直速度绝对值超过 `0.32 m/s` 起罚 |
| 14 | `dribbling_rapid_retouch_penalty` | -6.0 | -0.120 | 两次右脚 `<=14 N` 触球间隔 `<14 steps` 时惩罚 |
| 15 | `dribbling_useful_foot_touch` | +5.5 | +0.110 | 5 steps 后综合控制误差下降才奖励；CG 窗口作为软系数 |
| 16 | `dribbling_micro_contact_filter` | -4.0 | -0.080 | 右 ankle force EMA 超过 `22 N` 后连续惩罚，raw 上限 2 |
| 17 | `dribbling_undesired_contact_penalty` | -12.0 | -0.240 | 右 ankle 以外所有机器人 link 触球均惩罚，包括左脚 |
| 18 | `dribbling_cg_foot_ball_distance` | +3.5 | +0.070 | 仅在 CG reference foot--ball distance `<=0.55 m` 的接近/触球窗口跟踪，`std=0.22` |
| 19 | `action_rate_l2` | -0.1 | -0.002 | 对 effective action 做变化率惩罚 |
| 20 | `joint_limit` | -10.0 | -0.200 | soft joint limit 越界惩罚 |
| 21 | `upper_body_reference_overflow` | -0.05 | -0.001 | 上肢 reference 超出 action manifold 包络时惩罚 |

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

这些项不是权重设为 0，而是已从 Stage 2 配置及 `rewards_dribbling.py` 的对应旧实现中删除。S2 reward 数量由 28 降为 21。

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
| `ball_lost` | 50-step grace 后，pelvis--ball XY 距离 `>1.0 m` 或瞬时 XY 速度差 `>2.0 m/s` |
| `dribbling_no_contact` | 50-step grace 后，仅由右 ankle 接触清零；累计无接触达到 50 时结束。command-change recovery 与近球 recovery 可把计数增量降到 0.25，但都有有限预算 |
| `locomotion_manual_sequence_end` | play 使用 `--locomotion_cmd_reset_on_end` 且手动 command sequence 完成 |

本次没有修改 termination 或 reset。新的 recovery 闭环负责在 termination 之前真正缩短球距，而不是继续放宽失败条件。

## 7. Reset 与 command 行为

- control step 为 0.02 s；默认 episode 10 s。
- reset 时 robot 和 ball 一起重置，球位于 pelvis 前方约 `0.45 m`，robot/ball 初始速度清零。
- S2 的 motion clip 作为循环 style phase 使用，clip 结束不会单独重置场景。
- play 中 episode termination 会清空 RNN hidden state，但保持当前手动 speed/heading segment；`--locomotion_cmd_reset_on_end` 只在完整 sequence 结束时回到 segment 0。
- command 训练范围保持 speed `0.40--1.50 m/s`、heading `-0.75--0.75 rad`、duration `3--6 s`。
- 输入与 action 维度不变：Actor 163、Critic 295、Action 29，因此现有 S1/S2 checkpoint 在结构上兼容；本次不需要重训 S1。

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
- `useful_touch_new_event`、`useful_touch_evaluated`、`useful_touch_pending`
- `useful_touch_pre_error`、`useful_touch_current_error`、`useful_touch_improvement`
- `useful_touch_cg_aligned`、`useful_touch_reward`

旧 possession timer 和独立 CG contact-consistency telemetry 已删除，避免把不再参与当前 MDP 的状态误认为有效门控。

## 9. 下一次训练的验收重点

| 指标 | 目标 |
|---|---:|
| EMA 球速 MAE | `<=0.25 m/s` |
| `1.5 m/s` command 平均前向球速 | `1.25--1.75 m/s` |
| 球速 `>2.5 m/s` 比例 | `<5%` |
| 平均绝对 heading error | `<=0.12 rad` |
| 1800-step 实际失败 termination | `<=2` |
| 右脚接触间隔中位数 | 回到约 `18--24 steps`，同时 `<14 steps` 高频触球受控 |
| recovery 中 closing speed | 大部分为正，不再出现 chase reward 满分但距离持续增大 |
| useful touch 成功率 | 随训练上升；触球次数与有效触球必须分开看 |
| 手臂、腰部参考误差 | 至少保持 06 水平，并继续向 5.0 靠近 |

由于网络结构不变，可以从当前 Essay 08 的 S2 checkpoint warm resume；但 reward 语义发生了实质变化，需要新的 S2 训练适应。推荐保留旧 run 作为只读基线，使用新的 experiment/run 名称，避免混淆 diagnostic。
