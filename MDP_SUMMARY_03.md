# Footmimic MDP 03：夹球退化修复前后对照

本文以 `MDP_SUMMARY_01 copy.md` 记录的版本为基线，覆盖且只覆盖两个活动环境：

- Stage 1：`Tracking-CG-G1-Motion-RNN-mimic`
- Stage 2：`Tracking-CG-G1-Dribbling-RNN-control`

本次更新针对 `e02.npz` 暴露出的 Stage 2 夹球退化。重点不是重新堆叠历史 reward，
而是修复三个结构问题：近身状态仍能获得正奖励、间歇接触可以绕过连续接触惩罚、
以及球的净接触力与“最近 body”启发式无法可靠识别真正的接触 link。

## 1. 版本结论

| 项目 | `MDP_SUMMARY_01 copy.md` 基线 | 本次更新后 |
|---|---|---|
| Stage 1 input/reward/termination | 160 / 292 输入，11 reward，4 termination | **不变** |
| Stage 2 input/action | 160 / 292 输入，29 action | **不变，旧 checkpoint 输入层兼容** |
| Stage 2 reward 数量 | 26 | **27** |
| 安全走廊正奖励 | 走廊外仍有指数衰减正值 | **只在前向 `0.28–0.72 m` 内为正，走廊外严格为 0** |
| 过近/夹球约束 | 没有独立的连续位置惩罚 | **新增 `dribbling_ball_too_close_penalty`** |
| 持续接触判定 | 连续接触超过 3 steps | **20-step 接触 EMA / duty，无法用单帧断触清零** |
| 离脚滑球惩罚 | 离开接触后立即可能触发 | **真实触球后有 8-step release grace** |
| 接触 body 识别 | 球的合力 + 最近 body | **球—各机器人 link 的过滤接触力矩阵；旧 runtime 保留 fallback** |
| diagnostics | 有 reward 权重和一般球状态 | **增加接触 link、每-link 力、前向距离、duty、CG 错误类型等字段** |

RewardManager 仍以 `control_dt=0.02 s` 结算：

```text
实际每 control step 系数 = 配置 weight × 0.02
```

## 2. 原版本已经具备什么

### 2.1 两阶段一致的网络接口

基线已经完成 Stage 1 与 Stage 2 输入统一，本次没有改 observation term、顺序或维度。

#### Actor：160 维

| 顺序 | Term | 维度 | 含义 |
|---:|---|---:|---|
| 1 | `command` | 58 | demo joint position 29 + joint velocity 29 |
| 2 | `projected_gravity` | 3 | base frame 重力投影 |
| 3 | `motion_ref_ang_vel` | 3 | demo anchor angular velocity |
| 4 | `base_ang_vel` | 3 | robot base angular velocity |
| 5 | `joint_pos` | 29 | 相对默认姿态的关节位置 |
| 6 | `joint_vel` | 29 | 关节速度 |
| 7 | `actions` | 29 | 上一步实际执行的归一化 joint command |
| 8 | `anchor_ball_polar` | 3 | `[XY distance, cos(ball heading), sin(ball heading)]` |
| 9 | `motion_locomotion_polar_cmd` | 3 | `[speed, cos(command heading), sin(command heading)]` |

```text
58 + 3 + 3 + 3 + 29 + 29 + 29 + 3 + 3 = 160
```

#### Critic：292 维

| 顺序 | Term | 维度 | 含义 |
|---:|---|---:|---|
| 1 | `command` | 58 | demo joint position 29 + joint velocity 29 |
| 2 | `motion_anchor_pos_b` | 3 | reference anchor 相对 robot anchor 的位置 |
| 3 | `motion_anchor_ori_b` | 6 | reference anchor 相对姿态的 6D rotation |
| 4 | `body_pos` | 42 | 14 个 tracked bodies 的相对位置 |
| 5 | `body_ori` | 84 | 14 个 tracked bodies 的 6D rotation |
| 6 | `base_lin_vel` | 3 | robot base linear velocity |
| 7 | `base_ang_vel` | 3 | robot base angular velocity |
| 8 | `joint_pos` | 29 | 关节位置 |
| 9 | `joint_vel` | 29 | 关节速度 |
| 10 | `actions` | 29 | 上一步实际执行的归一化 joint command |
| 11 | `anchor_ball_polar` | 3 | task-frame 球极坐标 |
| 12 | `motion_locomotion_polar_cmd` | 3 | speed/heading command |

```text
58 + 3 + 6 + 42 + 84 + 3 + 3 + 29 + 29 + 29 + 3 + 3 = 292
```

Stage 1 的 polar command 来自 task-frame yaw 对齐后的 demo root XY velocity；Stage 2 来自
训练时重采样或 play 时手动给定的 speed/heading。`Hold duration=[3,6] s` 只决定训练 command
多久重采样一次，不进入网络，也不表示“保持不动”。Stage 2 训练 speed 仍为
`[0.40,1.50] m/s`，所以手动 `speed=0` 仍是泛化测试，不是受过专门训练的 STOP 行为。

### 2.2 Stage 1 已有的 MDP

Stage 1 的 11 个 reward 本次全部保留：

| Reward | Weight | 每步系数 | 作用 |
|---|---:|---:|---|
| `motion_body_pos` | `+1.0` | `+0.020` | locomotion bodies 相对位置 mimic |
| `motion_body_ori` | `+1.0` | `+0.020` | locomotion bodies 相对姿态 mimic |
| `motion_upper_body_pos` | `+0.85` | `+0.017` | 上肢相对位置 mimic |
| `motion_upper_body_ori` | `+0.35` | `+0.007` | 较弱的上肢姿态风格约束 |
| `motion_leg_lin_vel` | `+0.4` | `+0.008` | 腿部 yaw-aligned 线速度跟踪 |
| `motion_leg_ang_vel` | `+0.4` | `+0.008` | 腿部 yaw-aligned 角速度跟踪 |
| `motion_anchor_lin_vel` | `+2.2` | `+0.044` | demo pelvis/root 线速度跟踪 |
| `motion_anchor_pos_z` | `+0.6` | `+0.012` | demo anchor 高度跟踪 |
| `motion_foot_pos` | `+0.7` | `+0.014` | 双 ankle-roll 相对位置跟踪 |
| `action_rate_l2` | `-0.1` | `-0.002` | action 平滑 |
| `joint_limit` | `-10.0` | `-0.200` | soft joint limit |

Stage 1 termination 也不变：

| Termination | 条件 |
|---|---|
| `time_out` | 10 s，约 500 control steps |
| `anchor_pos_z` | `abs(z_ref-z_robot)>0.32 m` |
| `anchor_ori` | projected-gravity 的 Z 分量误差 `>0.8` |
| `ee_body_pos` | ankle-roll 或 wrist-yaw 的 Z 误差 `>0.25 m`，resample 后 grace 20 steps |

### 2.3 原 Stage 2 已有的能力

原 Stage 2 已经不是简单 mimic，而是 29 维 action 的 speed/heading controller：

- 用 `motion_anchor_lin_vel` 跟踪外部线速度命令。
- 用 heading error 生成有界 yaw-rate target，`motion_anchor_ang_vel` 不再以恒零 yaw rate 为目标。
- 保留 pelvis、足部和主要 body 的 demo 风格。
- 用 forward progress、proximity、chase、face-ball、speed cap、coast 和 orbit 项控制球。
- 用 CG foot-ball distance、CG contact consistency、合法脚触球、retouch、持续接触、
  接触力、弹跳和非法部位接触约束触球方式。
- 上肢 action 仍经过 rank-6 manifold、`1.8 Hz` low-pass 和 reference envelope，网络输出维度不变。

原版本的 26 项 reward 并非都无效；问题是其中若干项组合后，夹球策略获得的正回报仍然
大于能够真实触发的负回报。

## 3. `e02.npz` 暴露出的退化

`e02.npz` 共 331 个 diagnostic samples，记录的是更新前的 26-reward Stage 2。主要现象是：

| 指标 | `e02.npz` |
|---|---:|
| pelvis—ball XY 平均距离 | `0.1715 m` |
| pelvis—ball XY 中位数 | `0.1584 m` |
| 距离 `<0.28 m` | `93.35%` |
| 距离 `<0.32 m` | `96.07%` |
| 距离处于 `0.28–0.72 m` | `6.65%` |
| robot—ball contact rate | `32.02%` |
| 平均总 reward / control step | `0.3325` |

这说明球几乎全程停在人身近区，但策略的总 reward 并没有降低。部分关键 reward 的历史
平均值如下；diagnostic 中 `reward_terms` 是乘过配置 weight、尚未乘 `0.02` 的 term 值，
“实际/step”列再乘了 `0.02`。

| Reward | `e02` 加权均值 | 实际均值/step | 解释 |
|---|---:|---:|---|
| `dribbling_ball_forward_progress` | `+4.7712` | `+0.0954` | 夹球前进仍能稳定获得最大球任务正奖励 |
| `dribbling_cg_foot_ball_distance` | `+2.9878` | `+0.0598` | 球靠近脚时依然容易获得较高几何匹配奖励 |
| `dribbling_dynamic_proximity` | `+1.4264` | `+0.0285` | 虽然 93% 时间低于近边界，原实现仍持续给衰减正值 |
| `dribbling_ball_coast_penalty` | `-1.1657` | `-0.0233` | 会惩罚正常踢出后的离脚滑行，却不直接针对夹球位置 |
| `dribbling_sustained_contact_penalty` | `-0.3988` | `-0.0080` | 仅 6.65% samples 非零，连续计数容易被断触帧清零 |
| `dribbling_micro_contact_filter` | `-0.0479` | `-0.0010` | 只在 2.72% samples 非零 |
| `dribbling_ball_bounce_penalty` | `0` | `0` | 该 rollout 没有竖直弹跳，但该项仍是训练初期护栏 |
| `dribbling_undesired_contact_penalty` | `0` | `0` | 最近-body 启发式没有识别到夹球中的异常接触 |

因此退化不是“所有旧 penalty 都删错了”，也不是只需把历史白名单全部恢复。根因更具体：

1. `dynamic_proximity` 在近边界外仍给正值，导致错误状态没有明确的 reward 符号翻转。
2. 旧 `sustained_contact` 只看连续帧；夹球时接触力抖动即可绕过。
3. 球两侧同时受力可能在净力中相消；再用最近 body 猜接触者，会漏掉真实非法接触。
4. `coast_penalty` 没有考虑踢球后的自然 release，可能抑制本来需要学习的 kick-and-chase。

## 4. 本次代码更新

### 4.1 真实的逐-link 接触信号

球的 `ContactSensorCfg` 现在显式过滤 G1 的 30 个主要 rigid links，并读取
`force_matrix_w`。对每个环境分别得到球与各机器人 link 的接触力，而不是仅使用球所受合力。

当前接触标量定义为所有机器人 link 的水平接触力模长最大值：

```text
contact_force = max_link ||F_link_xy||
```

这样左右脚同时夹球时不会因合力方向相反而相消；接触 body 也由最大实际接触力决定，不再
依赖空间最近者。该信号同时被 contact reward、no-contact termination 和 diagnostics 使用。
若旧 IsaacLab runtime 没有 `force_matrix_w`，代码仍回退到原来的球净力与最近-body 路径，
但 diagnostics 会用 `ball_contact_filter_available=false` 明确标识 fallback。

受这一变化直接影响的项包括：

- `dribbling_legal_foot_touch`
- `dribbling_rapid_retouch_penalty`
- `dribbling_micro_contact_filter`
- `dribbling_undesired_contact_penalty`
- `dribbling_sustained_contact_penalty`
- `dribbling_cg_contact_consistency`
- `dribbling_ball_bounce_penalty`
- `dribbling_no_contact` termination

### 4.2 安全走廊正奖励改为硬区间门控

`dribbling_dynamic_proximity` 的 weight 仍为 `+3.0`，但函数语义改变：

```text
forward_offset < 0.28 m       -> reward = 0
0.28 m <= forward_offset <= 0.72 m -> lateral Gaussian reward
forward_offset > 0.72 m       -> reward = 0
```

也就是说，球靠得过近时不会再获得“虽然小但始终为正”的 proximity 信号。远端仍由 chase、
ball-lost 等机制处理，避免一个 reward 同时承担互相冲突的近端和远端塑形。

### 4.3 新增连续、接触无关的近身惩罚

新增 `dribbling_ball_too_close_penalty`：

```text
penalty = clip((0.28 - forward_offset) / (0.28 - 0.14), 0, 1)
weight  = -8.0
每步最大系数 = -0.160
```

它使用 command-frame 前向距离，不依赖接触传感器。球在 `0.28 m` 及以外时为 0；进入近区后
连续增大；到 `0.14 m` 或更近时达到满惩罚。即使接触力相消、接触短暂消失，夹在裆下或双脚
之间的球仍会被惩罚。

该项不是简单恢复旧 `dribbling_ball_trapped_penalty`。旧项依赖特定接触/双脚几何组合，
成熟轨迹只触发 2 steps；新项只依赖任务真正关心的 command-frame 球位置，信号稠密且可解释。

### 4.4 持续接触从“连续帧”改为“近期占空比”

`dribbling_sustained_contact_penalty` 保持 weight `-6.0`，但不再使用
`consecutive_contact_steps > 3`。现在维护 20-step EMA：

```text
alpha = 2 / (20 + 1)
duty_t = (1-alpha) * duty_(t-1) + alpha * contact_t
penalty = clip((duty_t - 0.25) / (0.60 - 0.25), 0, 1)
```

短促触球仍有容忍度；近期接触 duty 超过 25% 后开始连续惩罚，60% 时达到满惩罚。
夹球策略无法通过插入一个无接触 frame 把历史全部清零。

### 4.5 正常 release 获得 8-step 宽限

`dribbling_ball_coast_penalty` 保持 weight `-2.2`、近距和速度阈值不变，但真实接触后的前
8 control steps 不触发。50 Hz 下相当于约 `0.16 s`，让球从脚上自然释放后再进入 chase，
而不是刚踢出就立即被当作“无接触滑球”处罚。初始从未触球的高速近距滑球没有这段宽限。

## 5. 更新后 Stage 2 reward：27 项

### 5.1 基础 mimic / regularization：5 项

| Reward | Weight | 每步系数 | 状态 |
|---|---:|---:|---|
| `motion_anchor_lin_vel` | `+5.0` | `+0.100` | 不变，跟踪有效 locomotion 线速度命令 |
| `motion_foot_pos` | `+0.55` | `+0.011` | 不变，双脚 demo 相对位置 |
| `motion_body_pos` | `+0.45` | `+0.009` | 不变，主要 body demo 风格 |
| `action_rate_l2` | `-0.1` | `-0.002` | 不变，约束 effective action 变化 |
| `joint_limit` | `-10.0` | `-0.200` | 不变，soft joint limit |

### 5.2 身体与运动控制：6 项

| Reward | Weight | 每步系数 | 状态 |
|---|---:|---:|---|
| `motion_anchor_ang_vel` | `+1.0` | `+0.020` | 不变，跟踪 heading error 生成的 yaw-rate target |
| `locomotion_heading_tracking` | `+1.0` | `+0.020` | 不变，跟踪有效 heading |
| `dribbling_pelvis_quat_tracking` | `+2.0` | `+0.040` | 不变，pelvis demo 姿态 |
| `dribbling_gait_foot_tracking` | `+1.4` | `+0.028` | 不变，无接触时的 demo 足部步态 |
| `foot_distance` | `+0.35` | `+0.007` | 不变，双脚间距至少约 `0.24 m` |
| `upper_body_reference_overflow` | `-0.05` | `-0.001` | 不变，上肢 reference envelope |

Yaw-rate target 仍为：

```text
wz_target = clip(
    wz_feedforward + 2.0 * wrap(heading_effective - pelvis_yaw),
    -1.2,
    1.2,
)
```

### 5.3 球运动与位置：8 项

| Reward | Weight | 每步系数 | 本次状态与作用 |
|---|---:|---:|---|
| `dribbling_ball_forward_progress` | `+7.5` | `+0.150` | 不变；近期真实接触后才奖励沿 command 推进 |
| `dribbling_dynamic_proximity` | `+3.0` | `+0.060` | **修改**；只在前向 `0.28–0.72 m` 走廊给正值 |
| `dribbling_ball_too_close_penalty` | `-8.0` | `-0.160` | **新增**；前向 `<0.28 m` 连续惩罚，`<=0.14 m` 满惩罚 |
| `dribbling_chase_ball` | `+2.0` | `+0.040` | 不变；无接触、球在前方时奖励追球 |
| `dribbling_face_ball` | `+1.0` | `+0.020` | 不变；pelvis 朝 command 且球在前方 |
| `dribbling_ball_speed_excess` | `-2.5` | `-0.050` | 不变；球速超过 `1.35 m/s` 后惩罚 |
| `dribbling_ball_coast_penalty` | `-2.2` | `-0.044` | **修改**；触球后增加 8-step release grace |
| `dribbling_orbiting_penalty` | `-6.0` | `-0.120` | 不变；惩罚绕球横移 |

### 5.4 触球质量与安全：8 项

| Reward | Weight | 每步系数 | 本次状态与作用 |
|---|---:|---:|---|
| `dribbling_cg_foot_ball_distance` | `+3.5` | `+0.070` | 不变；CG 脚—球 XY 距离 |
| `dribbling_cg_contact_consistency` | `+1.0` | `+0.020` | 使用新的逐-link 接触信号 |
| `dribbling_legal_foot_touch` | `+5.5` | `+0.110` | 使用实际最大力 link 判断合法 ankle |
| `dribbling_rapid_retouch_penalty` | `-6.0` | `-0.120` | 使用新的逐-link 接触信号；间隔仍为 26 steps |
| `dribbling_sustained_contact_penalty` | `-6.0` | `-0.120` | **修改**；20-step duty EMA，25% 起罚，60% 满罚 |
| `dribbling_micro_contact_filter` | `-4.0` | `-0.080` | 使用真实 ankle 接触力；继续限制过强触球 |
| `dribbling_ball_bounce_penalty` | `-3.0` | `-0.060` | 接触时 `abs(vz)>0.32 m/s`；保留训练初期护栏 |
| `dribbling_undesired_contact_penalty` | `-12.0` | `-0.240` | 按实际接触 link 惩罚非合法 ankle，覆盖全部主要 G1 links |

总数为：

```text
5（基础）+ 6（身体/运动）+ 8（球运动/位置）+ 8（触球/安全）= 27
```

## 6. 更新后的 termination

Termination 项目和阈值没有增删；变化在于与接触相关的 termination 使用了更可靠的接触信号。

| Stage | Termination | 条件 | 本次变化 |
|---|---|---|---|
| Stage 1 | `time_out` | 10 s | 无 |
| Stage 1 | `anchor_pos_z` | Z error `>0.32 m` | 无 |
| Stage 1 | `anchor_ori` | gravity-Z error `>0.8` | 无 |
| Stage 1 | `ee_body_pos` | ankle/wrist Z error `>0.25 m`，grace 20 | 无 |
| Stage 2 | `time_out` | 10 s | 无 |
| Stage 2 | `ball_lost` | grace 50 后距离 `>1.0 m` 或速度差 `>2.0 m/s` | 无 |
| Stage 2 | `dribbling_no_contact` | grace 50 后无接触计数达到 50 | **接触由逐-link 最大力判断，减少力相消导致的误计数** |
| Stage 2 | `locomotion_manual_sequence_end` | manual sequence 最后一段结束且 `reset_on_end` | 无 |

`dribbling_no_contact` 的 75-step command-change recovery 仍保留：球距不超过 `0.85 m` 且
pelvis 至少以 `0.05 m/s` 接近球时，计数每步增加 `0.25`，真实接触清零。

## 7. diagnostics 新增内容

原版已经保存最终 runtime reward 配置：

- `reward_term_names`
- `reward_term_weights`：最终配置 weight
- `reward_term_step_weights`：`weight × control_dt`
- `reward_step_dt`
- `reward_terms`：每项的实际运行值
- `step_reward`

本次继续保留以上字段，并新增：

| 字段 | 内容 |
|---|---|
| `ball_command_forward_offset` | 球相对 pelvis 在 command 前向轴上的距离 |
| `ball_command_lateral_offset` | 球相对 pelvis 在 command 侧向轴上的偏移 |
| `ball_position_z` / `ball_vertical_speed` | 球高度与竖直速度 |
| `ball_contact_force` | 所有机器人 link 中最大的水平接触力 |
| `ball_contact_body_names` | 接触力矩阵各列对应的 G1 link 名称 |
| `ball_contact_body_force_magnitudes` | 每个 sample、每个 link 的球接触力模长 |
| `ball_contact_body_index` | 当前最大实际接触力的 link index；无可靠接触为 `-1` |
| `ball_contact_filter_available` | 是否实际获得逐-link 过滤接触矩阵 |
| `ball_undesired_body_contact` | 是否由非 ankle link 触球 |
| `ball_contact_duty_ema` | 20-step 接触占空比 EMA |
| `ball_contact_duty_penalty` | 持续接触 penalty 的原始 `[0,1]` 输出 |
| `ball_too_close_penalty` | 过近 penalty 的原始 `[0,1]` 输出 |
| `cg_label_available` | 当前 motion 是否有 CG contact 标注 |
| `cg_expected_contact` / `cg_expected_foot` | 当前 CG 期望接触及脚编号（0 左、1 右） |
| `cg_premature_contact` | CG 不应接触时发生接触 |
| `cg_missing_contact` | CG 应接触时没有实际接触 |
| `cg_wrong_foot_contact` | 接触存在但实际最大力 link 不是 CG 指定脚 |

play 结束时的控制台摘要也会输出 `too_close_rate`、平均 `contact_duty`、CG 提前接触率、
漏接触率和错脚率。三个 CG rate 分别以“期望无接触帧”“期望接触帧”和“期望且实际发生
接触帧”为分母。这样下一轮不只看总 reward，还能直接判断夹球究竟来自距离、接触持续性、
错误 body、错误时机还是错误脚。

## 8. 哪些旧内容仍不恢复

本次没有把所有历史 penalty 加回。以下判断保持不变：

- `target_destination_pos_local`、Cartesian 球坐标和 Cartesian locomotion command 继续删除；
  它们与当前 polar speed/heading 控制无关或重复。
- `motion_global_anchor_pos/ori`、固定 `+X` forward/lateral reward 继续删除；它们与可变 heading 冲突。
- `waist_action_rate_l2`、Stage 2 `motion_body_ori`、`pelvis_orientation` 等重复且低贡献项不恢复。
- `dribbling_cg_demo_ball_tracking` 在 `dribble_cg_use_demo_ball=False` 时结构性为零，不恢复。
- 旧 `dribbling_ball_trapped_penalty` 不恢复原实现；以新的、连续且接触无关的
  `dribbling_ball_too_close_penalty` 替代。

保留 `bounce`、`undesired contact` 和 `micro contact` 并不与上述清理矛盾：它们是针对训练
初期非法探索的护栏，成熟策略上趋近于零是成功约束后的正常结果。此次修复没有按单次成熟
diagnostic 的贡献率机械删项，而是区分“任务塑形项”和“应当在好策略上归零的安全约束”。

## 9. 预期效果与验证标准

本次改动预期先改变 reward landscape，再通过重新训练验证行为。旧 `model_55000.pt` 可以
用于接口和反事实 play，但它没有针对新 reward 训练，不能用一次旧模型播放直接判断最终效果。

建议新训练至少比较以下指标：

| 指标 | 期望方向 |
|---|---|
| `ball_command_forward_offset < 0.28 m` 比例 | 明显下降 |
| `0.28–0.72 m` 安全走廊占比 | 明显上升 |
| `ball_contact_duty_ema` | 避免长时间高于 0.25 |
| `ball_undesired_body_contact` | 接近 0 |
| `cg_premature_contact` / `cg_missing_contact` | 同时下降，而不是只优化一侧 |
| `cg_wrong_foot_contact` | 接近 0 |
| 球前向速度与 pelvis 速度 | 保持可控推进，而非以完全不触球换取安全 |
| `dribbling_ball_coast_penalty` | 正常 release 的早期 8 steps 不再触发 |

## 10. 对应代码

- Stage 1 / 共享 input / 球传感器：`source/whole_body_tracking/soccer/tasks/tracking/config/g1/soccer_flat_env_cfg.py`
- Stage 2 reward / termination 配置：`source/whole_body_tracking/soccer/tasks/tracking/config/g1/soccer_dribbling_env_cfg.py`
- 接触与控球 reward 实现：`source/whole_body_tracking/soccer/tasks/tracking/mdp/rewards_dribbling.py`
- termination 实现：`source/whole_body_tracking/soccer/tasks/tracking/mdp/terminations.py`
- diagnostics：`scripts/rsl_rl/play_multi.py`
