# Footmimic 当前版本 MDP：Input、Reward 与 Termination

本文记录当前两个注册环境中实际使用的配置。

- Stage 1：`Tracking-CG-G1-Motion-RNN-mimic`
- Stage 2：`Tracking-CG-G1-Dribbling-RNN-control`

## 1. 全局设置

| 项目 | 当前值 |
|---|---:|
| 仿真步长 | `0.005 s` |
| Control decimation | `4` |
| Policy/reward 周期 | `0.02 s` |
| 控制频率 | `50 Hz` |
| Episode 上限 | `10 s`，约 500 control steps |
| Actor observation | `160` 维 |
| Critic observation | `292` 维 |
| Policy action | `29` 维 |

RewardManager 每个 control step 实际加入总 reward 的系数为：

```text
每步系数 = 配置 weight × 0.02
```

## 2. Input

Stage 1 和 Stage 2 使用完全相同的 observation term、拼接顺序和维度。两者的主要区别仅是 locomotion command 的来源。

### 2.1 Actor observation：160 维

| 顺序 | Term | 维度 | 含义 | 训练噪声 |
|---:|---|---:|---|---|
| 1 | `command` | 58 | demo joint position 29 + joint velocity 29 | 无 |
| 2 | `projected_gravity` | 3 | base frame 中的重力投影 | `[-0.05, 0.05]` |
| 3 | `motion_ref_ang_vel` | 3 | demo anchor angular velocity | `[-0.05, 0.05]` |
| 4 | `base_ang_vel` | 3 | robot base angular velocity | `[-0.2, 0.2]` |
| 5 | `joint_pos` | 29 | 相对默认姿态的关节位置 | `[-0.01, 0.01]` |
| 6 | `joint_vel` | 29 | 关节速度 | `[-0.5, 0.5]` |
| 7 | `actions` | 29 | 上一步实际执行的归一化 joint command | 无 |
| 8 | `anchor_ball_polar` | 3 | pelvis 到球的 task-frame 平面极坐标 `[XY distance, cos(heading), sin(heading)]` | 无 |
| 9 | `motion_locomotion_polar_cmd` | 3 | `[speed, cos(heading), sin(heading)]` | 无 |

维度校验：

```text
58 + 3 + 3 + 3 + 29 + 29 + 29 + 3 + 3 = 160
```

其中 `actions` 是实际执行的 action：

- Stage 1：普通 29-joint position action。
- Stage 2：经过 upper-body manifold 投影和低通滤波后的 effective action。

### 2.2 Critic observation：292 维

| 顺序 | Term | 维度 | 含义 |
|---:|---|---:|---|
| 1 | `command` | 58 | demo joint position 29 + joint velocity 29 |
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

维度校验：

```text
58 + 3 + 6 + 42 + 84 + 3 + 3 + 29 + 29 + 29 + 3 + 3 = 292
```

Critic observation 不添加 policy observation noise。

### 2.3 两个阶段的 command 来源

#### Stage 1 Mimic

- `locomotion_command_mode="reference"`。
- Polar locomotion command 来自经过 task-frame yaw 对齐的 demo root XY velocity。

#### Stage 2 Control

- 训练时 `locomotion_command_mode="resampled"`。
- Play 时可以切换为 `manual`。
- 输入来自经过限速和平滑处理的外部 speed/heading command，而不是 demo root velocity。

Stage 2 当前训练 command 范围：

| 参数 | 当前值 |
|---|---:|
| Speed | `[0.40, 1.50] m/s` |
| Heading | `[-0.75, 0.75] rad` |
| Hold duration | `[3.0, 6.0] s` |
| Yaw-rate command | `0 rad/s` |
| Heading slew limit | `0.85 rad/s` |
| Acceleration limit | `1.4 m/s²` |
| Deceleration limit | `2.4 m/s²` |
| Turn slowdown threshold | `0.55 rad` |
| Minimum turn speed scale | `0.60` |

`Hold duration` 只控制 command 重新采样的间隔，不进入网络。当前训练 speed 下限是 `0.40 m/s`，因此 `speed=0` 只属于手动测试，不是单独训练过的 STOP 状态。

## 3. Reward

### 3.1 Stage 1 Mimic：11 项

| Reward | Weight | 每步系数 | 当前作用 |
|---|---:|---:|---|
| `motion_body_pos` | `+1.0` | `+0.020` | 跟踪 pelvis、左右 hip-roll/knee 和 torso 的 demo 相对位置 |
| `motion_body_ori` | `+1.0` | `+0.020` | 跟踪上述 locomotion bodies 的 demo 相对姿态 |
| `motion_upper_body_pos` | `+0.85` | `+0.017` | 跟踪左右 shoulder/elbow/wrist 的相对位置 |
| `motion_upper_body_ori` | `+0.35` | `+0.007` | 较弱地保持上肢姿态风格 |
| `motion_leg_lin_vel` | `+0.4` | `+0.008` | 跟踪 yaw-aligned 腿部线速度 |
| `motion_leg_ang_vel` | `+0.4` | `+0.008` | 跟踪 yaw-aligned 腿部角速度 |
| `motion_anchor_lin_vel` | `+2.2` | `+0.044` | 跟踪 demo pelvis/root 线速度 |
| `motion_anchor_pos_z` | `+0.6` | `+0.012` | 跟踪 demo anchor 高度 |
| `motion_foot_pos` | `+0.7` | `+0.014` | 跟踪左右 ankle-roll 相对位置 |
| `action_rate_l2` | `-0.1` | `-0.002` | 惩罚相邻 action 的平方变化 |
| `joint_limit` | `-10.0` | `-0.200` | 惩罚关节超过 soft position limit |

Stage 1 不使用控球推进 reward，主要学习动作、步态和触球风格，为 Stage 2 提供初始化。

### 3.2 Stage 2 Control：26 项

Stage 2 的 reward 可以先按来源分成两层：

- **沿用 Stage 1 的同名 reward：5 项。** 保留基础动作风格、速度、足部位置和安全正则，但部分目标、body 范围或实现会根据 Control 任务调整。
- **Stage 2 新增 reward：21 项。** 再按功能分为身体/运动控制 6 项、球运动与位置控制 7 项、触球质量与安全约束 8 项。

因此总数为：

```text
5（沿用 Stage 1）+ 6（身体/运动控制）+ 7（球运动/位置）+ 8（触球/安全）= 26
```

#### 3.2.1 沿用 Stage 1 的同名 reward：5 项

这些 reward 在两个阶段都有，但 Stage 2 使用自己的权重和 Control 语义。例如 `motion_anchor_lin_vel` 从跟踪 demo root velocity 改为跟踪外部 locomotion command；`action_rate_l2` 约束的是 manifold/filter 后真正执行的 action。

| Reward | Weight | 每步系数 | 当前作用 |
|---|---:|---:|---|
| `motion_anchor_lin_vel` | `+5.0` | `+0.100` | 跟踪当前有效 locomotion 线速度命令 |
| `motion_foot_pos` | `+0.55` | `+0.011` | 跟踪双脚 demo 相对位置 |
| `motion_body_pos` | `+0.45` | `+0.009` | 较弱地保留主要 body 的 demo 相对位置风格 |
| `action_rate_l2` | `-0.1` | `-0.002` | 惩罚 manifold/filter 后实际执行 action 的变化 |
| `joint_limit` | `-10.0` | `-0.200` | 惩罚关节超过 soft position limit |

#### 3.2.2 Stage 2 新增：身体与运动控制，6 项

这组 reward 负责把 Stage 1 的动作模仿能力转换成可按外部 speed/heading command 控制的运动，同时维持 pelvis、双脚和上肢的合理姿态。

| Reward | Weight | 每步系数 | 当前作用 |
|---|---:|---:|---|
| `motion_anchor_ang_vel` | `+1.0` | `+0.020` | 跟踪转向所需的 pelvis yaw-rate target |
| `locomotion_heading_tracking` | `+1.0` | `+0.020` | 跟踪平滑后的有效 heading |
| `dribbling_pelvis_quat_tracking` | `+2.0` | `+0.040` | 保留 demo pelvis 姿态风格 |
| `dribbling_gait_foot_tracking` | `+1.4` | `+0.028` | 无球接触期间跟踪 demo 双脚步态 |
| `foot_distance` | `+0.35` | `+0.007` | 鼓励双脚间距至少 `0.24 m` |
| `upper_body_reference_overflow` | `-0.05` | `-0.001` | 惩罚上肢目标超出 demo reference `±0.25 rad` envelope |

#### 3.2.3 Stage 2 新增：球运动与位置控制，7 项

这组 reward 决定球应该往哪里走、走多快，以及机器人相对球应该处于什么位置。它们主要塑造“沿 command 带球前进、不要丢球、不要绕球或放任球滑行”的行为。

| Reward | Weight | 每步系数 | 当前作用 |
|---|---:|---:|---|
| `dribbling_ball_forward_progress` | `+7.5` | `+0.150` | 奖励球沿当前 command heading 前进，并限制过大的侧向运动 |
| `dribbling_dynamic_proximity` | `+3.0` | `+0.060` | 将球保持在 command-frame 前方 `0.28–0.72 m` 的安全走廊 |
| `dribbling_chase_ball` | `+2.0` | `+0.040` | 无接触且球在 command 前方时奖励追球速度 |
| `dribbling_face_ball` | `+1.0` | `+0.020` | 鼓励 pelvis 朝向 command，并使球位于 command 前方 |
| `dribbling_ball_speed_excess` | `-2.5` | `-0.050` | 球 XY 速度超过 `1.35 m/s` 后进行惩罚 |
| `dribbling_ball_coast_penalty` | `-2.2` | `-0.044` | 惩罚近距离、无接触的高速滑球 |
| `dribbling_orbiting_penalty` | `-6.0` | `-0.120` | 惩罚围绕球横向绕行而不沿 command 推进 |

#### 3.2.4 Stage 2 新增：触球质量与安全约束，8 项

这组 reward 负责决定“如何触球”：触球脚和接触时机应符合 CG 示范，只允许合法 ankle 触球，并抑制过密、持续、过强、弹跳或其他身体部位触球。

| Reward | Weight | 每步系数 | 当前作用 |
|---|---:|---:|---|
| `dribbling_cg_foot_ball_distance` | `+3.5` | `+0.070` | 匹配 CG 合成轨迹中的触球脚—球 XY 距离 |
| `dribbling_cg_contact_consistency` | `+1.0` | `+0.020` | 使实际接触与 CG 标注接触相符 |
| `dribbling_legal_foot_touch` | `+5.5` | `+0.110` | 奖励合法 ankle 接触；默认左右脚均合法 |
| `dribbling_rapid_retouch_penalty` | `-6.0` | `-0.120` | 两次合法轻触间隔小于 26 steps 时惩罚 |
| `dribbling_sustained_contact_penalty` | `-6.0` | `-0.120` | 连续接触超过 3 steps 后惩罚持续推球 |
| `dribbling_micro_contact_filter` | `-4.0` | `-0.080` | 惩罚过强的合法 ankle 接触 |
| `dribbling_ball_bounce_penalty` | `-3.0` | `-0.060` | 触球期间 `abs(vz)>0.32 m/s` 时惩罚竖直弹球 |
| `dribbling_undesired_contact_penalty` | `-12.0` | `-0.240` | 球接触的最近 body 不是合法 ankle 时惩罚 |

`motion_anchor_ang_vel` 使用的 yaw-rate target 为：

```text
wz_target = clip(
    wz_feedforward + 2.0 * wrap(heading_effective - pelvis_yaw),
    -1.2,
    1.2,
)
```

其中 `wz_feedforward` 来自 heading 平滑过程；pelvis heading 落后时，比例项负责追赶。

Stage 2 reward 的总体目标是：

1. 按指定速度和方向运动。
2. 让球沿指定方向推进。
3. 保持适当的人球距离并及时追球。
4. 使用合法、短促且不过强的脚部接触。
5. 保留 demo 的步态和姿态风格。

## 4. Termination

### 4.1 Stage 1 Mimic

| Termination | 类型 | 当前条件 |
|---|---|---|
| `time_out` | timeout | Episode 达到 `10 s`，约 500 control steps |
| `anchor_pos_z` | failure | `abs(z_ref - z_robot) > 0.32 m` |
| `anchor_ori` | failure | reference 与 robot projected-gravity 的 Z 分量误差 `> 0.8` |
| `ee_body_pos` | failure | ankle-roll 或 wrist-yaw 的 Z 误差 `> 0.25 m`；motion resample 后有 20 steps grace |

### 4.2 Stage 2 Control

| Termination | 类型 | 当前条件 |
|---|---|---|
| `time_out` | timeout | Episode 达到 `10 s`，约 500 control steps |
| `ball_lost` | failure | 前 50 steps grace 后，人球 XY 距离 `>1.0 m`，或者二者 XY 速度差 `>2.0 m/s` |
| `dribbling_no_contact` | failure | 前 50 steps grace 后，无接触计数达到 50 |
| `locomotion_manual_sequence_end` | play/reset | 仅 manual play 启用 `reset_on_end` 时，在最后一段 command 结束后触发 |

`dribbling_no_contact` 带有限恢复窗口：command 改变后的 75 steps 内，如果球距离不超过 `0.85 m`，并且 pelvis 以至少 `0.05 m/s` 接近球，则无接触计数每步只增加 `0.25`；真实接触会将计数清零。

## 5. 当前代码位置

- 共享 observation、基础 reward 和 timeout：`source/whole_body_tracking/soccer/tasks/tracking/tracking_env_cfg.py`
- Stage 1 input、reward 和 termination：`source/whole_body_tracking/soccer/tasks/tracking/config/g1/soccer_flat_env_cfg.py`
- Stage 2 input、reward 和 termination：`source/whole_body_tracking/soccer/tasks/tracking/config/g1/soccer_dribbling_env_cfg.py`
- Reward 实现：`source/whole_body_tracking/soccer/tasks/tracking/mdp/rewards.py`、`source/whole_body_tracking/soccer/tasks/tracking/mdp/rewards_dribbling.py`
- Termination 实现：`source/whole_body_tracking/soccer/tasks/tracking/mdp/terminations.py`
