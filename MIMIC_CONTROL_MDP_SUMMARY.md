# Footmimic v5.0 清理后 MDP 总结

> 本文保留“夹球退化修复之前”的清理结果。当前 27-reward MDP、前后差异与新 diagnostics
> 请以 [`MDP_SUMMARY_03.md`](MDP_SUMMARY_03.md) 为准。

本文以当时清理后的代码为准，只覆盖两个仍注册的环境：

- `Tracking-CG-G1-Motion-RNN-mimic`：Stage 1 Mimic。
- `Tracking-CG-G1-Dribbling-RNN-control`：Stage 2 Control。

历史环境、别名和未注册变体不在本文范围内。当前仿真步长为 `0.005 s`，控制 decimation
为 4，因此 policy/reward 更新周期 `control_dt=0.02 s`，控制频率为 50 Hz。RewardManager
实际加入每步总 reward 的系数为 `配置 weight × 0.02`。

## 1. 清理后的输入接口

### 1.1 总维度

| 项目 | Stage 1 Mimic | Stage 2 Control |
|---|---:|---:|
| Actor observation | 160 | 160 |
| Critic observation | 292 | 292 |
| Policy action | 29 | 29 |
| Episode 上限 | 10 s，约 500 control steps | 10 s，约 500 control steps |
| 根速度目标来源 | demo reference | 外部 speed/heading command |

两个阶段的 term 名称、拼接顺序和维度完全相同，可以使用相同的神经网络输入层。

### 1.2 Actor observation：160 维

以下顺序就是进入 actor 的拼接顺序。

| 顺序 | Term | 维度 | 坐标系/含义 | Policy 训练噪声 |
|---:|---|---:|---|---|
| 1 | `command` | 58 | demo joint position 29 + joint velocity 29 | 无 |
| 2 | `projected_gravity` | 3 | base frame 重力投影 | `[-0.05, 0.05]` |
| 3 | `motion_ref_ang_vel` | 3 | demo anchor angular velocity | `[-0.05, 0.05]` |
| 4 | `base_ang_vel` | 3 | robot base angular velocity | `[-0.2, 0.2]` |
| 5 | `joint_pos` | 29 | 相对默认姿态的关节位置 | `[-0.01, 0.01]` |
| 6 | `joint_vel` | 29 | 关节速度 | `[-0.5, 0.5]` |
| 7 | `actions` | 29 | 上一步实际执行的归一化 joint command | 无 |
| 8 | `anchor_ball_polar` | 3 | 球的 task-frame 平面极坐标 `[distance, cos(heading), sin(heading)]` | 无 |
| 9 | `motion_locomotion_polar_cmd` | 3 | `[speed, cos(heading), sin(heading)]` | 无 |

维度校验：基础 policy observation 为 154 维，polar 球坐标和 polar locomotion command
各 3 维，总计 `154+3+3 = 160`。

球的位置只通过 `anchor_ball_polar` 输入网络。这里的 task frame 是与 world 平行的环境局部
`+X/+Y` 平面；距离是 pelvis 到球的 XY 距离。速度/方向命令同样只保留 polar 表达。

`actions` 在两个阶段调用同一个 `effective_joint_action` 接口：Stage 1 没有特殊后处理时回退
为 raw action；Stage 2 返回 upper-body manifold 投影和低通滤波后实际执行的 action。

### 1.3 Critic observation：292 维

| 顺序 | Term | 维度 | 含义 |
|---:|---|---:|---|
| 1 | `command` | 58 | demo joint position + velocity |
| 2 | `motion_anchor_pos_b` | 3 | reference anchor 相对 robot anchor 的位置 |
| 3 | `motion_anchor_ori_b` | 6 | reference anchor 相对姿态的 6D rotation 表达 |
| 4 | `body_pos` | 42 | 14 个 tracked bodies 的相对位置 |
| 5 | `body_ori` | 84 | 14 个 tracked bodies 的 6D rotation 表达 |
| 6 | `base_lin_vel` | 3 | robot base linear velocity |
| 7 | `base_ang_vel` | 3 | robot base angular velocity |
| 8 | `joint_pos` | 29 | 关节位置 |
| 9 | `joint_vel` | 29 | 关节速度 |
| 10 | `actions` | 29 | 上一步实际执行的归一化 joint command |
| 11 | `anchor_ball_polar` | 3 | 球的 task-frame 平面极坐标 |
| 12 | `motion_locomotion_polar_cmd` | 3 | 当前 speed/heading command |

维度校验：基础 privileged observation 为 286 维，polar 球坐标和 polar command 共 6 维，
总计 `286+6 = 292`。Critic 输入不添加 policy observation noise。

### 1.4 两个阶段的输入差异只在数据来源

- Stage 1 的 locomotion command mode 为 `reference`。Polar command 来自经过 task-frame
  yaw 对齐的 demo root XY velocity。
- Stage 2 的 mode 为 `resampled`；play 时可切换为 `manual`。输入来自经过限速和平滑处理的
  外部 speed/heading command，而不是 demo root velocity。
- 当前 Control 没有 `IDLE/DRIBBLE/STOP` one-hot 输入。`speed=0` 只是零速控制命令，
  不代表单独训练过的 STOP 状态。

Stage 2 的训练 command 范围如下：

| 参数 | 当前值 |
|---|---:|
| Speed | `[0.40, 1.50] m/s` |
| Heading | `[-0.75, 0.75] rad` |
| Hold duration | `[3.0, 6.0] s` |
| Yaw-rate target | `0 rad/s` |
| Heading slew limit | `0.85 rad/s` |
| Acceleration / deceleration limit | `1.4 / 2.4 m/s²` |
| Turn slowdown threshold | `0.55 rad` |
| Minimum turn speed scale | `0.60` |

`Hold duration` 只是每组 speed/heading command 在重新采样前持续的时间，不进入网络输入，
也不表示保持静止。当前训练 speed 下限为 `0.40 m/s`，因此零速只可用于手动测试，不能视为
已经学习过的 STOP 行为。

### 1.5 已删除的 input

| 已删除项 | 原维度 | 删除原因 |
|---|---:|---|
| `target_destination_pos_local` | 3 | 当前 speed/heading control 没有 destination reward 或 termination；随机目的地与 locomotion command 无绑定，只会提供无关位置特征。其采样、状态和 marker 已一并删除。 |
| `motion_anchor_lin_vel_cmd` | 3 | Control 中等于 `[speed·cos(heading), speed·sin(heading), 0]`，可由保留的 polar command 精确恢复，属于重复输入。 |
| `motion_anchor_ang_vel_cmd` | 3 | 当前 Control 的 `wz_range=(0,0)`，训练中恒为零；Stage 1 又已有 `motion_ref_ang_vel`，没有必要保留第二套角速度命令。 |
| `target_point_pos` | 3 | 与 `anchor_ball_polar` 重复描述同一个球；同时输入 Cartesian 和 polar 会增加输入复杂度并产生重复特征。 |
| `anchor_ball_local` | 3 | Cartesian 球坐标实现已无调用；统一只保留 polar 版本后直接删除函数。 |
| `locomotion_task_state` | 3 | 活动 Control 已没有 IDLE/DRIBBLE/STOP 状态机，保留 one-hot 会制造没有对应训练语义的输入。 |
| target first-frame/cache/blind-zone/positional-encoding 观察函数 | 可变 | 当前两个环境均未引用，是历史目标点实验遗留实现。 |
| `target_destination_pos_local_first_frame` | 3 | 历史目的地实验遗留；当前任务完全不使用 destination。 |
| `foot_target_point_distance` | 可变 | 未进入 actor/critic，也没有活动配置调用。 |
| 未接入的 raw anchor observation 辅助函数 | 可变 | 已被当前明确的 reference command 和 privileged terms 替代。 |

Checkpoint loader 会按 term 语义选择旧列，而不是从尾部盲目截断：旧 Stage 1 的
`163/295`、上一版统一接口的 `169/301` 和旧 Stage 2 的 `172/304` 都会删除 Cartesian 球、
destination、Cartesian linear command 和恒零 angular command；旧 Stage 1 缺少的 polar
locomotion command 以零权重初始化，其余列和 normalizer 统计保持原顺序迁移。

## 2. Stage 1 Mimic rewards

Stage 1 当前共有 11 个 reward。下面的权重是运行时最终配置值。

| Reward | Weight | 每步系数 | 当前作用 |
|---|---:|---:|---|
| `motion_body_pos` | `+1.0` | `+0.020` | 跟踪 pelvis、左右 hip-roll/knee 和 torso 的 demo 相对位置。 |
| `motion_body_ori` | `+1.0` | `+0.020` | 跟踪上述 locomotion bodies 的 demo 相对姿态。 |
| `motion_upper_body_pos` | `+0.85` | `+0.017` | 跟踪左右 shoulder/elbow/wrist 的相对位置。 |
| `motion_upper_body_ori` | `+0.35` | `+0.007` | 较弱地保持上肢姿态风格，避免过度约束转向。 |
| `motion_leg_lin_vel` | `+0.4` | `+0.008` | 跟踪 yaw-aligned 腿部线速度。 |
| `motion_leg_ang_vel` | `+0.4` | `+0.008` | 跟踪 yaw-aligned 腿部角速度。 |
| `motion_anchor_lin_vel` | `+2.2` | `+0.044` | 跟踪 demo pelvis/root 线速度。 |
| `motion_anchor_pos_z` | `+0.6` | `+0.012` | 只跟踪 demo anchor 高度。 |
| `motion_foot_pos` | `+0.7` | `+0.014` | 跟踪左右 ankle-roll 相对位置。 |
| `action_rate_l2` | `-0.1` | `-0.002` | 惩罚相邻 action 的平方变化，原始值最大裁剪到 100。 |
| `joint_limit` | `-10.0` | `-0.200` | 惩罚全部关节超过 soft position limit。 |

Stage 1 不创建球控制 reward；它只学习动作、步态和触球风格，为 Stage 2 提供初始化。

## 3. Stage 2 Control rewards

Stage 2 当前共有 26 个 reward。`诊断占比`来自
`diagnostic_20260802_133050.npz` 的 2718 个 control steps，表示各项平均加权 reward 的
绝对贡献占比。该诊断在 input 和 reward 清理前录制；函数与参数未改变的项仍可用作历史
参考。`motion_anchor_ang_vel` 的目标生成方式已经改变，因此旧诊断值不再适用于它；三个
恢复的安全 penalty 则保留旧诊断值，以说明成熟策略上的历史触发情况，而不是预测重新训练后的贡献。

| Reward | Weight | 每步系数 | 诊断均值/step | 诊断占比 | 当前作用 |
|---|---:|---:|---:|---:|---|
| `dribbling_ball_forward_progress` | `+7.5` | `+0.150` | `+0.076621` | `17.65%` | 奖励球沿当前 command heading 前进，并限制过大的侧向运动。 |
| `motion_anchor_lin_vel` | `+5.0` | `+0.100` | `+0.056389` | `12.99%` | 跟踪当前有效的三维 locomotion 线速度命令。 |
| `dribbling_cg_foot_ball_distance` | `+3.5` | `+0.070` | `+0.043783` | `10.08%` | 匹配 CG 合成轨迹中的触球脚—球 XY 距离。 |
| `dribbling_chase_ball` | `+2.0` | `+0.040` | `+0.033045` | `7.61%` | 无接触且球在 command 前方时奖励追球速度。 |
| `dribbling_pelvis_quat_tracking` | `+2.0` | `+0.040` | `+0.030884` | `7.11%` | 保留 demo pelvis 姿态风格。 |
| `dribbling_ball_speed_excess` | `-2.5` | `-0.050` | `-0.022823` | `5.26%` | 球 XY 速度超过 `1.35 m/s` 后线性惩罚。 |
| `dribbling_face_ball` | `+1.0` | `+0.020` | `+0.019561` | `4.51%` | 同时要求球位于 command 前方且 pelvis 朝向 command。 |
| `dribbling_orbiting_penalty` | `-6.0` | `-0.120` | `-0.019463` | `4.48%` | 惩罚围绕球横向绕行而不沿 command 推进。 |
| `locomotion_heading_tracking` | `+1.0` | `+0.020` | `+0.018380` | `4.23%` | 跟踪平滑后的有效 heading；低速时按 command speed 门控。 |
| `dribbling_cg_contact_consistency` | `+1.0` | `+0.020` | `+0.013893` | `3.20%` | 使实际接触与 CG 标注接触相符。 |
| `motion_anchor_ang_vel` | `+1.0` | `+0.020` | 不适用（函数已改） | 不适用 | 跟踪由 heading 平滑器前馈和 pelvis heading error 共同生成的 yaw-rate target；`gain=2.0`，限幅 `±1.2 rad/s`，`std=1.0`。 |
| `action_rate_l2` | `-0.1` | `-0.002` | `-0.013642` | `3.14%` | 惩罚实际执行 action 的变化，包含 manifold/filter 后结果。 |
| `dribbling_dynamic_proximity` | `+3.0` | `+0.060` | `+0.013120` | `3.02%` | 将球保持在 command-frame 前方 `0.28–0.72 m` 的安全走廊。 |
| `dribbling_rapid_retouch_penalty` | `-6.0` | `-0.120` | `-0.008300` | `1.91%` | 两次合法轻触间隔少于 26 steps 时惩罚。 |
| `dribbling_legal_foot_touch` | `+5.5` | `+0.110` | `+0.008054` | `1.85%` | 奖励合法 ankle 接触，默认左右脚均合法。 |
| `upper_body_reference_overflow` | `-0.05` | `-0.001` | `-0.007546` | `1.74%` | 惩罚上肢目标超出 demo reference `±0.25 rad` envelope。 |
| `foot_distance` | `+0.35` | `+0.007` | `+0.006763` | `1.56%` | 鼓励双脚间距至少 `0.24 m`，减少交叉和重叠。 |
| `dribbling_gait_foot_tracking` | `+1.4` | `+0.028` | `+0.005071` | `1.17%` | 无球接触期间跟踪 demo 双脚步态。 |
| `motion_body_pos` | `+0.45` | `+0.009` | `+0.004746` | `1.09%` | 以较弱权重保留 12 个主要 body 的 demo 相对位置风格。 |
| `joint_limit` | `-10.0` | `-0.200` | `-0.004044` | `0.93%` | 惩罚全部关节超过 soft position limit。 |
| `dribbling_ball_coast_penalty` | `-2.2` | `-0.044` | `-0.003009` | `0.69%` | 球在 pelvis `0.50 m` 内、无接触且速度超过 `0.55 m/s` 时惩罚滑行。 |
| `motion_foot_pos` | `+0.55` | `+0.011` | `+0.002740` | `0.63%` | 跟踪双脚 demo 相对位置。 |
| `dribbling_sustained_contact_penalty` | `-6.0` | `-0.120` | `-0.002208` | `0.51%` | 连续接触超过 3 steps 后惩罚，防止持续推球。 |
| `dribbling_micro_contact_filter` | `-4.0` | `-0.080` | `-0.000229` | `0.05%` | 对合法 ankle 的接触力做 EMA；超过 `22` 后连续惩罚过强触球，单项原始输出最大为 `2.0`。 |
| `dribbling_ball_bounce_penalty` | `-3.0` | `-0.060` | `0` | `0%` | 球接触期间 `abs(vz)>0.32 m/s` 时惩罚，堵住只检查 XY 球速留下的竖直弹球漏洞。 |
| `dribbling_undesired_contact_penalty` | `-12.0` | `-0.240` | `0` | `0%` | 球接触最近 body 不是合法 ankle 时惩罚，防止通过膝盖或手腕触球获取推进 reward。 |

`motion_anchor_ang_vel` 现在只比较 pelvis 的实际 yaw rate。目标为：

`wz_target = clip(wz_feedforward + 2.0 * wrap(heading_effective - pelvis_yaw), -1.2, 1.2)`

其中 `wz_feedforward` 根据最终 heading 与平滑后有效 heading 的误差生成，并受 command 的
`0.85 rad/s` heading rate limit 限制。正在转向时它提供非零前馈，pelvis 落后时比例项负责
追赶；最终 heading、有效 heading 和 pelvis yaw 对齐后，目标才回到零。

从该诊断看，最主要的正向信号是球前进、根线速度、CG 脚球距离、追球和 pelvis 姿态；
最主要的负向信号是球速过高、绕球和 action rate。`dribbling_sustained_contact_penalty`
虽然很弱，但仍略高于本次采用的 `0.5%` 清理阈值，因此保留。三个恢复项在成熟模型诊断中
贡献很低，但它们属于训练初期的行为护栏：约束成功后趋近于零并不能证明训练时没有作用。

## 4. 已删除的 rewards 及原因

### 4.1 最终配置中无训练信号或结构性无效

这些项不是简单从列表隐藏，而是已从活动配置删除；没有剩余调用的项目，其实现也已删除。

| 已删除 Reward | 原状态 | 删除原因 |
|---|---:|---|
| `motion_global_anchor_pos` | weight `0` | 没有训练梯度；全局 XY tracking 还会限制 task-frame command 控制。 |
| `motion_global_anchor_ori` | weight `0` | 没有训练梯度；与当前 task-frame yaw 对齐策略不一致。 |
| `motion_body_lin_vel` | weight `0` | 没有训练梯度，并与保留的 anchor/腿部速度跟踪重复。 |
| `motion_body_ang_vel` | weight `0` | 没有训练梯度，并与保留的角速度/姿态跟踪重复。 |
| `target_point_proximity` | weight `0` | 已由 command-frame `dribbling_dynamic_proximity` 取代。 |
| `task_heading_alignment` | weight `0` | 已由 `locomotion_heading_tracking` 和 `dribbling_face_ball` 覆盖。 |
| `forward_velocity` | weight `0` | 固定 `+X` 目标与可变 heading control 冲突。 |
| `lateral_velocity_penalty` | weight `0` | 固定横向惩罚会错误压制斜向 command。 |
| `dribbling_velocity_tracking` | weight `0` | 与 `motion_anchor_lin_vel`、球前进和追球项重复。 |
| `dribbling_approach_foot_ball` | weight `0` | CG foot-ball distance 已覆盖接近球的几何关系。 |
| `dribbling_cg_premature_contact` | weight `0` | 没有训练信号，且与 contact consistency/retouch 约束重叠。 |
| `dribbling_cg_demo_ball_tracking` | weight `+4.0`，实际恒为 0 | 当前 `dribble_cg_use_demo_ball=False`，函数在该结构下始终返回 0，属于结构性无效项。 |

### 4.2 仍删除的低贡献项

以下判断使用同一份 2718-step v5.0 diagnostics。占比为绝对加权贡献占比。

| 已删除 Reward | 原 weight | 原诊断占比 | 删除原因 |
|---|---:|---:|---|
| `waist_action_rate_l2` | `-0.25` | `0.4979%` | 低于阈值，且与全关节 `action_rate_l2` 重复。 |
| `pelvis_orientation` | `-2.5` | `0.27%` | 已有 `dribbling_pelvis_quat_tracking`。 |
| `dribbling_support_ankle_roll` | `+0.25` | `0.23%` | 稀疏且只作用于少量标注帧。 |
| `motion_body_ori`（仅 Stage 2） | `+0.35` | `0.17%` | Control 中贡献很低；pelvis 姿态已有专门项。Stage 1 仍保留。 |
| `dribbling_cg_foot_consistency` | `+0.5` | `0.08%` | 与 CG contact consistency 和 foot-ball distance 重叠。 |
| `dribbling_stall_no_touch_penalty` | `-5.5` | `0.07%` | 2718 steps 中只触发 8 steps。 |
| `upper_body_manifold_nullspace` | `-0.02` | `0.06%` | 每步计算但数值极弱，无法证明对策略有有效约束。 |
| `dribbling_ball_trapped_penalty` | `-8.0` | `0.03%` | 只触发 2 steps。 |
| `undesired_contacts` | `-0.1` | `<0.01%` | 只触发 13 steps，且与更明确的触球合法性逻辑重叠。 |

`dribbling_ball_bounce_penalty`、`dribbling_undesired_contact_penalty` 和
`dribbling_micro_contact_filter` 已从本节移回活动 reward 表。它们不是按成熟轨迹贡献筛选的
任务塑形项，而是分别约束竖直弹球、非法部位触球和过强 ankle 接触的早期训练护栏。

## 5. Terminations

### 5.1 Stage 1 Mimic

| Termination | 类型 | 当前条件 | 目的 |
|---|---|---|---|
| `time_out` | timeout | 10 s，约 500 control steps | 固定 episode 上限。 |
| `anchor_pos_z` | failure | `abs(z_ref-z_robot) > 0.32 m` | 只阻止明显高度崩溃，不限制水平任务轨迹。 |
| `anchor_ori` | failure | reference 与 robot projected-gravity 的 Z 分量误差 `>0.8` | 终止严重倾倒/姿态失配。 |
| `ee_body_pos` | failure | ankle-roll 或 wrist-yaw 的 Z 误差 `>0.25 m`；resample 后 20 steps grace | 阻止末端高度严重偏离，同时给重采样后的恢复时间。 |

### 5.2 Stage 2 Control

| Termination | 类型 | 当前条件 | 目的 |
|---|---|---|---|
| `time_out` | timeout | 10 s，约 500 control steps | 固定训练 episode 上限。 |
| `ball_lost` | failure | 前 50 steps grace 后，球—pelvis XY 距离 `>1.0 m`，或二者 XY 速度差 `>2.0 m/s` | 结束已经失去控球的 rollout。 |
| `dribbling_no_contact` | failure | 前 50 steps grace 后，无接触计数达到 50 | 防止只在球附近摆姿态但一直不触球。 |
| `locomotion_manual_sequence_end` | play/reset | 仅 manual play 设置 `reset_on_end` 且最后一段结束时触发 | 重置 robot、ball 和手动命令序列；正常 resampled 训练不触发。 |

`dribbling_no_contact` 在 command 改变后的 75-step recovery window 内，如果球距不超过
`0.85 m` 且 pelvis 至少以 `0.05 m/s` 接近球，无接触计数每步只增加 `0.25`；真实接触会
把计数清零。这是有限恢复窗口，不允许无限期无接触存活。

### 5.3 从活动环境删除或不再接入的 termination

| 项目 | 清理后的状态 | 原因 |
|---|---|---|
| `anchor_pos_z`、`anchor_ori`、`ee_body_pos`（Stage 2） | 只在 Stage 1 创建 | 它们衡量 demo mimic 偏差；command-control 不应因偏离某一 demo frame 而结束。 |
| `motion_finished` | 两阶段均未接入 | demo clip 结束不等于 task episode 结束，避免把控制任务重新退化为动作回放。 |
| `dribbling_stop_success` | 未接入活动 Control | 当前没有 IDLE/DRIBBLE/STOP 状态机，也没有训练独立 STOP success。 |
| `contact_phase_violation` | 未接入 | 属于旧 contact-phase 实验，当前使用连续 contact reward 和 no-contact timeout。 |
| `interaction_termination` | 未接入 | 属于已经删除的历史 interaction 环境。 |
| 全身 `bad_anchor_pos` / `bad_motion_body_pos` | 未接入 | Stage 1 使用更宽松的 Z-only guards，允许水平 task-frame 对齐和转向。 |

这里“未接入”表示它不属于当前两个环境的 TerminationManager；部分通用函数仍保留在
`terminations.py`，用于兼容工具或后续实验，并不在训练/播放时执行。

## 6. Action 接口

- 两阶段 policy 均输出 29 维 normalized joint-position action，因此 Stage 1 可以作为
  Stage 2 warm start。
- Stage 1 使用普通 29-joint position action。
- Stage 2 保持 29 维网络接口，但对 14 个上肢关节应用 rank-6 PCA manifold、
  `1.8 Hz` low-pass、reference `±0.25 rad` envelope 和 orthogonal residual limit；
  其余关节走普通 joint-position action 路径。
- Stage 2 的 reward 与 observation 使用后处理后的 effective action，避免网络看到或奖励
  一个实际上没有送入机器人执行的 action。

## 7. Diagnostics 与实际 reward weight

运行 `play_multi.py --diagnostic` 会在当前 run 的 `diagnostics/` 目录生成 `.npz`。清理后的
diagnostics 包含：

- `reward_term_names`
- `reward_term_weights`：RewardManager 初始化后的最终配置 weight
- `reward_term_step_weights`：`weight × control_dt`
- `reward_step_dt`
- `reward_terms`：每项已经加权并乘 dt 的逐步 reward
- `step_reward`
- action、joint tracking、command、ball、contact、manifold、torque 和 termination telemetry

因此新 diagnostics 可以直接确认运行时实际启用的 reward 数量、名称和最终权重，不需要再从
配置文件反推。清理后的 Stage 2 文件应只包含本文列出的 26 个 reward term。

## 8. 当前代码入口

- Stage 1/共享输入：[`soccer_flat_env_cfg.py`](source/whole_body_tracking/soccer/tasks/tracking/config/g1/soccer_flat_env_cfg.py)
- Stage 2：[`soccer_dribbling_env_cfg.py`](source/whole_body_tracking/soccer/tasks/tracking/config/g1/soccer_dribbling_env_cfg.py)
- 基础 input/reward/termination：[`tracking_env_cfg.py`](source/whole_body_tracking/soccer/tasks/tracking/tracking_env_cfg.py)
- Reward 实现：[`rewards.py`](source/whole_body_tracking/soccer/tasks/tracking/mdp/rewards.py)、[`rewards_dribbling.py`](source/whole_body_tracking/soccer/tasks/tracking/mdp/rewards_dribbling.py)
- Termination 实现：[`terminations.py`](source/whole_body_tracking/soccer/tasks/tracking/mdp/terminations.py)
- Checkpoint 输入迁移：[`checkpoint_loading.py`](source/whole_body_tracking/soccer/utils/checkpoint_loading.py)
- Play/diagnostics：[`play_multi.py`](scripts/rsl_rl/play_multi.py)
