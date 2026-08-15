# Footmimic MDP 04：v5 接触语义与逐-link 防夹球双通道

本文由 `MDP_SUMMARY_03.md` 更新而来，只覆盖当前保留的两个环境：

- Stage 1：`Tracking-CG-G1-Motion-RNN-mimic`
- Stage 2：`Tracking-CG-G1-Dribbling-RNN-control`

04 版不增加、不删除 reward，也不改变 observation、action、reward weight 或 termination 阈值。
本次只修正 03 版把逐-link 最大力替换成全局接触信号后造成的语义漂移：

- v5 已验证过的全局 reward/termination 恢复使用球的净接触力。
- 防夹球、非法身体接触和接触 link 识别继续使用逐-link 接触力。
- diagnostics 同时保存两路信号，不再让 `ball_contact_force` 同时表示两种物理量。

## 1. 从 03 到 04 的变化

| 项目 | 03 版 | 04 版 |
|---|---|---|
| 网络输入 | Stage 1/2 均为 actor 160、critic 292 | 不变 |
| action | 29 维 joint action | 不变 |
| Stage 1 reward / termination | 11 / 4 | 不变 |
| Stage 2 reward / termination | 27 / 4 | 不变 |
| 全局接触标量 | `max_link ||F_link,xy||` | 恢复 v5：`||F_net,xy||` |
| 防夹球接触标量 | 与全局信号共用 | 独立使用 `max_link ||F_link,xy||` |
| `cg_contact_consistency`、`orbiting_penalty` | 公式和权重未变，但输入信号偏离 v5 | 公式、权重和接触输入均回到 v5 语义 |
| no-contact termination | max-link 接触 | 恢复 v5 净力接触 |
| diagnostics | `ball_contact_force` 实际表示 max-link 力 | `ball_contact_force` 表示净力，新增显式 max-link 字段 |

RewardManager 的控制周期仍为 `control_dt = 0.02 s`：

```text
实际每 control step 系数 = 配置 weight × 0.02
```

## 2. 双接触通道

### 2.1 全局净力通道：兼容 v5 行为

```text
global_contact_force = ||F_ball_net,xy||
global_contact = global_contact_force > threshold
```

该信号继续使用 `soccer_ball_contact` 的 3 帧 history，与 v5 的
`soccer_ball_contact_force_magnitude()` 定义一致。它允许不同接触的反向力在球的净力中抵消，
因此不适合单独识别夹球，但适合恢复以 v5 数据标定的正常触球节奏和门控。

使用全局通道的逻辑包括：

- `_dribbling_sim_contact` 及其调用者
- `dribbling_cg_contact_consistency`
- `dribbling_orbiting_penalty` 的 weak-contact gate
- `dribbling_ball_forward_progress` 的 recent-contact gate
- `dribbling_dynamic_proximity` 的 no-contact damping
- `dribbling_ball_coast_penalty` 的 release grace
- `dribbling_ball_bounce_penalty`
- `dribbling_gait_foot_tracking`、`dribbling_chase_ball`
- `dribbling_no_contact` termination
- diagnostics 的 `ball_contact` / `ball_contact_force`

### 2.2 逐-link 通道：防夹球和接触安全

```text
link_contact_force = max_link ||F_ball<-link,xy||
link_contact = link_contact_force > threshold
```

`ContactSensorCfg.filter_prim_paths_expr` 继续覆盖 G1 的 30 个主要 rigid links，
从 `force_matrix_w` 读取球与每个机器人 link 的接触力。左右两侧同时挤压球时，即使净力接近零，
该通道仍能检测到接触。

使用逐-link 通道的逻辑包括：

- `dribbling_legal_foot_touch`
- `dribbling_rapid_retouch_penalty`
- `dribbling_micro_contact_filter`
- `dribbling_undesired_contact_penalty`
- `dribbling_sustained_contact_penalty` 的 20-step duty EMA
- diagnostics 的接触 body、每-link 力及非法 body 判断

旧 IsaacLab runtime 若没有 `force_matrix_w`，逐-link 通道会回退到全局净力；此时环境仍可运行，
但 diagnostics 的 `ball_contact_filter_available=false` 会表明无法可靠区分接触 link。

## 3. Input 与 action

04 版没有修改网络接口。03 版 checkpoint 保持兼容；原始 v5.0 的 172/304 输入 checkpoint
不能直接视为当前 160/292 网络的同构 checkpoint，本次目标是恢复类似行为效果，而不是直接复用其输入层。

### 3.1 Actor：160 维

| 顺序 | Term | 维度 | 含义 |
|---:|---|---:|---|
| 1 | `command` | 58 | demo joint position 29 + joint velocity 29 |
| 2 | `projected_gravity` | 3 | base-frame 重力投影 |
| 3 | `motion_ref_ang_vel` | 3 | demo anchor angular velocity |
| 4 | `base_ang_vel` | 3 | robot base angular velocity |
| 5 | `joint_pos` | 29 | 相对默认姿态的关节位置 |
| 6 | `joint_vel` | 29 | 关节速度 |
| 7 | `actions` | 29 | 上一步实际执行的归一化 joint command |
| 8 | `anchor_ball_polar` | 3 | `[XY distance, cos(ball heading), sin(ball heading)]` |
| 9 | `motion_locomotion_polar_cmd` | 3 | `[speed, cos(command heading), sin(command heading)]` |

### 3.2 Critic：292 维

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

两阶段 action 都是 29 维。Stage 2 的 yaw-rate target 仍由 heading error 生成：

```text
wz_target = clip(
    wz_feedforward + 2.0 * wrap(heading_effective - pelvis_yaw),
    -1.2,
    1.2,
)
```

`Hold duration=[3,6] s` 只决定训练 command 的重采样间隔，不进入网络，也不表示保持不动。
Stage 2 的训练 speed 仍为 `[0.40,1.50] m/s`，所以手动 `speed=0` 是泛化测试。

## 4. Stage 1 MDP

Stage 1 本次完全不变。

### 4.1 Reward：11 项

| Reward | Weight | 每步系数 | 作用 |
|---|---:|---:|---|
| `motion_body_pos` | +1.0 | +0.020 | locomotion bodies 相对位置 mimic |
| `motion_body_ori` | +1.0 | +0.020 | locomotion bodies 相对姿态 mimic |
| `motion_upper_body_pos` | +0.85 | +0.017 | 上肢相对位置 mimic |
| `motion_upper_body_ori` | +0.35 | +0.007 | 较弱的上肢姿态约束 |
| `motion_leg_lin_vel` | +0.4 | +0.008 | 腿部 yaw-aligned 线速度跟踪 |
| `motion_leg_ang_vel` | +0.4 | +0.008 | 腿部 yaw-aligned 角速度跟踪 |
| `motion_anchor_lin_vel` | +2.2 | +0.044 | demo pelvis/root 线速度跟踪 |
| `motion_anchor_pos_z` | +0.6 | +0.012 | demo anchor 高度跟踪 |
| `motion_foot_pos` | +0.7 | +0.014 | 双 ankle-roll 相对位置跟踪 |
| `action_rate_l2` | -0.1 | -0.002 | action 平滑 |
| `joint_limit` | -10.0 | -0.200 | soft joint limit |

### 4.2 Termination：4 项

| Termination | 条件 |
|---|---|
| `time_out` | 10 s，约 500 control steps |
| `anchor_pos_z` | `abs(z_ref-z_robot) > 0.32 m` |
| `anchor_ori` | projected-gravity Z 分量误差 `> 0.8` |
| `ee_body_pos` | ankle/wrist Z 误差 `> 0.25 m`，resample grace 20 steps |

## 5. Stage 2 Reward：27 项

本次没有增加 reward，也没有调整任何 weight。下表中的“接触通道”用于明确 04 版的实际输入语义。

### 5.1 基础 mimic / regularization：5 项

| Reward | Weight | 每步系数 | 接触通道 / 作用 |
|---|---:|---:|---|
| `motion_anchor_lin_vel` | +5.0 | +0.100 | 无；跟踪有效 locomotion 线速度命令 |
| `motion_foot_pos` | +0.55 | +0.011 | 无；双脚 demo 相对位置 |
| `motion_body_pos` | +0.45 | +0.009 | 无；主要 body demo 风格 |
| `action_rate_l2` | -0.1 | -0.002 | 无；effective action 平滑 |
| `joint_limit` | -10.0 | -0.200 | 无；soft joint limit |

### 5.2 身体与运动控制：6 项

| Reward | Weight | 每步系数 | 接触通道 / 作用 |
|---|---:|---:|---|
| `motion_anchor_ang_vel` | +1.0 | +0.020 | 无；heading-error yaw-rate target |
| `locomotion_heading_tracking` | +1.0 | +0.020 | 无；有效 heading 跟踪 |
| `dribbling_pelvis_quat_tracking` | +2.0 | +0.040 | 无；pelvis demo 姿态 |
| `dribbling_gait_foot_tracking` | +1.4 | +0.028 | **全局**；无接触时保持 demo 足部步态 |
| `foot_distance` | +0.35 | +0.007 | 无；双脚距离至少约 0.24 m |
| `upper_body_reference_overflow` | -0.05 | -0.001 | 无；上肢 reference envelope |

### 5.3 球运动与位置：8 项

| Reward | Weight | 每步系数 | 接触通道 / 作用 |
|---|---:|---:|---|
| `dribbling_ball_forward_progress` | +7.5 | +0.150 | **全局**；recent-contact 后沿 command 推进 |
| `dribbling_dynamic_proximity` | +3.0 | +0.060 | **全局门控**；仅前向 0.28–0.72 m 走廊给正值 |
| `dribbling_ball_too_close_penalty` | -8.0 | -0.160 | 几何；前向 `<0.28 m` 连续惩罚，`<=0.14 m` 满惩罚 |
| `dribbling_chase_ball` | +2.0 | +0.040 | **全局**；无接触且球在前方时追球 |
| `dribbling_face_ball` | +1.0 | +0.020 | 几何；pelvis 面向 command 且球在前方 |
| `dribbling_ball_speed_excess` | -2.5 | -0.050 | 速度；球速超过 1.35 m/s 后惩罚 |
| `dribbling_ball_coast_penalty` | -2.2 | -0.044 | **全局**；真实触球后保留 8-step release grace |
| `dribbling_orbiting_penalty` | -6.0 | -0.120 | **全局**；使用 v5 weak-contact gate 惩罚绕球横移 |

### 5.4 触球质量与安全：8 项

| Reward | Weight | 每步系数 | 接触通道 / 作用 |
|---|---:|---:|---|
| `dribbling_cg_foot_ball_distance` | +3.5 | +0.070 | 几何；CG 脚—球 XY 距离 |
| `dribbling_cg_contact_consistency` | +1.0 | +0.020 | **全局**；恢复 v5 的 CG 接触一致性语义 |
| `dribbling_legal_foot_touch` | +5.5 | +0.110 | **逐-link**；奖励实际 ankle 的新 gentle touch |
| `dribbling_rapid_retouch_penalty` | -6.0 | -0.120 | **逐-link**；合法触球间隔小于 26 steps 时惩罚 |
| `dribbling_sustained_contact_penalty` | -6.0 | -0.120 | **逐-link**；20-step duty EMA，25% 起罚、60% 满罚 |
| `dribbling_micro_contact_filter` | -4.0 | -0.080 | **逐-link**；限制 ankle 过强触球 |
| `dribbling_ball_bounce_penalty` | -3.0 | -0.060 | **全局**；接触时 `abs(vz)>0.32 m/s` |
| `dribbling_undesired_contact_penalty` | -12.0 | -0.240 | **逐-link**；非 ankle link 触球 |

## 6. Stage 2 Termination：4 项

| Termination | 条件 | 04 接触语义 |
|---|---|---|
| `time_out` | 10 s | 无接触输入 |
| `ball_lost` | grace 50 后距离 `>1.0 m` 或速度差 `>2.0 m/s` | 无接触输入 |
| `dribbling_no_contact` | grace 50 后无接触计数达到 50 | **全局净力，恢复 v5 语义** |
| `locomotion_manual_sequence_end` | manual sequence 结束且 `reset_on_end` | 无接触输入 |

`dribbling_no_contact` 的 75-step command-change recovery 保留：球距不超过 0.85 m，且 pelvis
至少以 0.05 m/s 接近球时，计数每步增加 0.25；全局接触会清零计数。

## 7. Diagnostics

### 7.1 Reward 实际配置

diagnostic 继续保存：

- `reward_term_names`
- `reward_term_weights`：最终 runtime 配置 weight
- `reward_term_step_weights`：`weight × control_dt`
- `reward_step_dt`
- `reward_terms`：每项实际运行值
- `step_reward`

### 7.2 双接触字段

| 字段 | 04 版含义 |
|---|---|
| `ball_contact_force` | **全局净水平接触力**，与 v5 reward/termination 门控一致 |
| `ball_contact` | `ball_contact_force > 1.0` |
| `ball_max_link_contact_force` | **所有过滤机器人 link 中最大的水平接触力** |
| `ball_link_contact` | `ball_max_link_contact_force > 1.0` |
| `ball_contact_body_names` | 逐-link 力矩阵列对应的 G1 link 名称 |
| `ball_contact_body_force_magnitudes` | 每 sample、每 link 的水平接触力模长 |
| `ball_contact_body_index` | 当前最大逐-link 力对应的 link index；无可靠接触为 -1 |
| `ball_contact_filter_available` | runtime 是否提供逐-link `force_matrix_w` |
| `ball_undesired_body_contact` | 按逐-link 通道判断是否由非 ankle link 触球 |
| `ball_contact_duty_ema` | 按逐-link 通道计算的 20-step 接触 duty EMA |
| `ball_contact_duty_penalty` | sustained-contact penalty 原始 `[0,1]` 输出 |

`cg_premature_contact`、`cg_missing_contact` 与 `cg_contact_consistency` 一致，按全局通道统计；
`cg_wrong_foot_contact` 仍结合逐-link body 识别判断实际触球脚。

play 结束时同时输出：

```text
contact_rate       = 全局净力接触率
link_contact_rate  = 逐-link 接触率
```

两者的差值本身就是诊断信息：如果 `link_contact_rate` 明显高于 `contact_rate`，说明存在较多
相互抵消、多 link 同时接触或净力很弱但局部受力明显的情况。

## 8. 仍然删除的内容及原因

| 删除内容 | 删除原因 |
|---|---|
| `target_destination_pos_local` | 当前持续运球控制不使用终点位置，且与 polar locomotion 接口无关 |
| Cartesian 球坐标 | 与 `anchor_ball_polar` 重复，统一只保留极坐标 |
| Cartesian locomotion command observation | 与 `motion_locomotion_polar_cmd` 重复 |
| `motion_anchor_lin_vel_cmd` / `motion_anchor_ang_vel_cmd` observation | 与 polar command 重复；注意同名 tracking reward 没有删除 |
| `motion_global_anchor_pos/ori` | 与可变 heading 的局部 task frame 冲突 |
| 固定世界 `+X` forward/lateral reward | 转向时任务几何错误 |
| `waist_action_rate_l2` | 与 effective `action_rate_l2` 重复 |
| Stage 2 `motion_body_ori` / `pelvis_orientation` 等 | 与现有姿态/heading 项重复且贡献低 |
| `dribbling_cg_demo_ball_tracking` | `dribble_cg_use_demo_ball=False` 时结构性为零 |
| 旧 `dribbling_ball_trapped_penalty` | 特定双脚几何且成熟轨迹仅触发约 2 steps；由连续位置和逐-link duty 约束替代 |

本次没有因为成熟 rollout 上贡献接近零而删除 `bounce`、`undesired_contact` 或
`micro_contact_filter`。这些是训练初期非法探索的护栏，目标是在成熟策略上归零。

## 9. 本轮预期与快速验证

本轮不是单项 ablation，而是一次有明确因果关系的接触语义修复包。预期同时满足：

1. 保持 03 已实现的防夹球能力。
2. `cg_contact_consistency`、recent-contact、orbit/no-contact gates 回到 v5 的标定范围。
3. 降低正常 kick-release 被误判为持续接触或无接触失败的概率。
4. 恢复转向速度、球前进速度和触球节奏，同时不重新引入 e02 的近身夹球。

下一次 diagnostic 应优先比较：

- `contact_rate` 与 `link_contact_rate`
- `<0.28 m` 过近比例与 0.28–0.72 m 安全走廊比例
- ball / pelvis command-forward speed
- mean absolute heading error
- no-contact 与 ball-lost termination 数量
- sustained duty、undesired body contact、CG missing/premature/wrong-foot
- 每项 reward 的实际加权贡献

## 10. 对应代码

- 共享 input / 球传感器：`source/whole_body_tracking/soccer/tasks/tracking/config/g1/soccer_flat_env_cfg.py`
- Stage 2 reward / termination 配置：`source/whole_body_tracking/soccer/tasks/tracking/config/g1/soccer_dribbling_env_cfg.py`
- 双接触通道与 reward：`source/whole_body_tracking/soccer/tasks/tracking/mdp/rewards_dribbling.py`
- termination：`source/whole_body_tracking/soccer/tasks/tracking/mdp/terminations.py`
- diagnostics：`scripts/rsl_rl/play_multi.py`

