# S2 Reward 6.0.1（仓库实现快照）

本文只记录仓库提交 `07bb021`（`6.1`）中
`G1FlatCGDribblingUnifiedS2LocalReferenceEnvCfg` 的实际最终配置，不混入后续版本的实验设计。

- Gym task：`Tracking-CG-G1-Dribbling-RNN-unified-s2-local-reference`
- 默认 reference：`motions/master-single/master_001675_001884_unitree_g1.npz`
- 有效 reward：共 **19 项**，其中基础 motion / safety reward 13 项，S2 专用 reward 6 项
- 核心目标：保留 reference 动作与 pelvis-local twist，同时在 reference 指定的时间、脚和 inside/outside 脚面完成真实物理触球
- 球完全由物理仿真驱动；reward 不读取 reference 球位置、reference 球速度或 contact-to-contact ball flow
- 表中的权重是 `RewardTermCfg.weight`；Isaac Lab 在汇总单步 reward 时还会乘环境步长 `dt=0.02`

## 1. 最终 Reward 总表

### 1.1 Motion / safety reward（13 项）

| Reward term | 权重 | 作用 |
|---|---:|---|
| `motion_body_pos` | `+0.72` | 跟踪指定 body 的相对位置，保留 reference 全身结构。 |
| `motion_body_ori` | `+1.0` | 跟踪 pelvis、torso 和上肢等 body 的 reference 朝向。 |
| `motion_body_lin_vel` | `+0.3` | 轻量跟踪 yaw 对齐后的 reference body 线速度。 |
| `motion_body_ang_vel` | `+0.3` | 轻量跟踪 yaw 对齐后的 reference body 角速度。 |
| `motion_anchor_lin_vel` | `+5.0` | 跟踪 reference pelvis-local `[vx, vy]`，`std=0.60`。 |
| `motion_anchor_ang_vel` | `+1.0` | 跟踪 reference pelvis yaw rate，`std=1.50`。 |
| `action_rate_l2` | `-0.1` | 惩罚有效执行动作的突变。 |
| `waist_action_rate_l2` | `-0.25` | 额外抑制 waist yaw/roll/pitch 抖动。 |
| `joint_limit` | `-10.0` | 惩罚超过 soft joint limits 的关节位置。 |
| `undesired_contacts` | `-0.1` | 惩罚普通环境中的膝、躯干等非法接触。 |
| `pelvis_orientation` | `-2.5` | 抑制 pelvis 侧倾、后仰和失稳。 |
| `upper_body_reference_overflow` | `-0.05` | 惩罚超出上肢 reference 包络的 action target。 |
| `upper_body_manifold_nullspace` | `-0.02` | 惩罚会被上肢 PCA 投影丢弃的 action 分量。 |

### 1.2 S2 专用 reward（6 项）

| Reward term | 权重 | 输出 | 作用 |
|---|---:|---|---|
| `s2_windowed_foot_tracking` | `+1.0` | 连续 `[0,1]` | 平时跟踪双脚 reference；事件前逐渐释放指定触球脚，支撑脚始终保持完整跟踪。 |
| `s2_contact_proximity` | `+0.5` | 连续 `[0,1]` | 在事件前和接触窗口内，引导指定脚靠近物理球并进入标注的 inside/outside 一侧。 |
| `s2_valid_contact_bonus` | `+4.5` | 每事件一次脉冲 | 奖励在正确窗口内由指定脚、指定脚面完成的新物理触球。 |
| `s2_nonfoot_ball_contact` | `-1.0` | 新接触脉冲 | 惩罚膝、手腕、身体等非脚踝 link 碰球。 |
| `s2_wrong_foot_contact` | `-0.25` | 新接触脉冲 | 惩罚 contact window 内错误脚触球。 |
| `s2_premature_contact` | `-0.25` | 每目标至多一次 | 惩罚目标 contact window 之前发生的脚踝触球。 |

S2 的触球部分为：

```text
R_s2 = 1.0  * windowed_foot_tracking
     + 0.5  * contact_proximity
     + 4.5  * valid_contact
     - 1.0  * nonfoot_ball_contact
     - 0.25 * wrong_foot_contact
     - 0.25 * premature_contact
```

## 2. Windowed Foot Tracking

`s2_windowed_foot_tracking` 使用双脚相对位置误差：

```text
r_foot = exp(-mean(weight_foot * ||p_ref - p_sim||^2) / 0.30^2)
```

- 支撑脚权重始终为 `1.0`；
- 指定触球脚从事件前 `0.30 s` 开始由 `1.0` 线性降到 `0.10`；
- contact window 内，指定触球脚继续使用 `0.10`；
- window 外恢复完整 reference 脚跟踪。

它的作用是让非触球阶段继续模仿 reference gait，同时在触球附近给指定脚足够自由度接近物理球。

## 3. Contact Proximity

### 3.1 实际脚球几何

proximity 使用**实际指定脚**的世界位置和 yaw，而不是 reference 球轨迹。令球在实际脚 yaw-local 坐标中的偏移为：

```text
ball_from_foot = [x_foot, y_foot, z_foot]
physical_distance = ||ball_from_foot||
```

物理可达误差：

```text
reach_error = relu(physical_distance - 0.25)
```

inside/outside 标签先按照左右脚转换成统一的 medial 坐标，再计算指定脚面侧误差：

```text
side_error = relu(0.04 - signed_lateral)
```

因此球在指定脚面正确一侧超过 `0.04 m` 后，`side_error=0`。最终区域距离为：

```text
target_distance = sqrt(reach_error^2 + side_error^2)
```

### 3.2 6.1 的前向硬 gate

6.1 使用脚 yaw-local 的 `x_foot` 构造线性 gate：

```text
front_gate = clip((x_foot - (-0.05)) / (0.05 - (-0.05)), 0, 1)
```

- `x_foot <= -0.05 m`：`front_gate=0`；
- `x_foot >= +0.05 m`：`front_gate=1`；
- 中间区间线性变化。

proximity 的空间部分为：

```text
region_score = 1 / (1 + (target_distance / 0.12)^2)
```

最终：

```text
contact_proximity = region_score * front_gate * active * time_weight
```

### 3.3 生效时间

`contact_proximity` 不是只在发生物理接触时生效。其时间范围为：

```text
active = 当前 reference contact window
      OR 下一事件前 0.30 s
```

事件前 `0.30 s` 内，`time_weight` 从 `0.20` 线性增加到 `1.0`；进入 contact window 后保持 `1.0`。当前事件第一次有效触球成功后，proximity 对该事件关闭，防止持续贴球刷分。

## 4. 有效触球判定

`s2_valid_contact_bonus` 必须同时满足：

1. 当前位于 reference contact window，中心前后各 `0.10 s`；
2. 当前步出现新的物理接触上升沿；
3. 接触 body 是合法脚踝 link；
4. 接触脚与 reference 指定脚一致；
5. inside/outside 标签有效；
6. 物理球位于指定脚面的正确一侧，并越过 `0.04 m` dead zone；
7. 当前事件之前尚未成功。

触球方向和球速不参与有效性判定。触球传感器保留约 `1 N` 的噪声下限，但没有 `150 N` hard gate、soft-force reward 或 force termination。

每个事件最多支付一次：

```text
4.5 * 1 = 4.5          # RewardTermCfg 加权值
4.5 * dt = 0.09        # dt=0.02 时进入 PPO 的实际单次贡献
```

## 5. 错误接触

三个惩罚与正奖励共享同一个 `dribbling_s2_contact_event_state()`，因此对新接触、接触 body、指定脚和时间窗口的判断一致。

- `s2_nonfoot_ball_contact`：新的非脚踝 robot-ball 接触；
- `s2_wrong_foot_contact`：窗口内由另一只脚触球；
- `s2_premature_contact`：下一目标窗口之前发生脚踝触球，每个目标事件最多计一次。

错误脚、错误脚面或过早接触都不能设置该事件成功。`s2_touch_occurrence` 只用于 telemetry，不进入 reward。

## 6. 时序覆盖与已知限制

6.1 的 S2 球信号时序为：

```text
事件前 0.30 s                 contact window               下一事件前 0.30 s
       |---------------------------|                              |
       contact_proximity + valid_contact                          contact_proximity
```

第一次有效触球之后，如果距离下一事件仍超过 `0.30 s`，这段物理过渡没有专门的球可达性 reward。第一次触球只要满足当前事件的时刻、脚和脚面，就立即得到完整 `+4.5`；球是否因此变得更容易被下一脚触到，不影响当前奖励。

这是 6.1 的实现事实，也是后续设计需要解决的主要 credit-assignment 空窗。

## 7. Termination 与 Curriculum

- `ball_lost`：球与 pelvis 的 XY 距离连续超过 `1.10 m` 才终止；不使用速度发散条件。
- `dribbling_no_contact`：S2 中关闭。
- reference 脚踝/手腕高度 termination：S2 中关闭。
- `missed_valid_contact`：事件窗口结束并经过 `3` 个 grace step 后仍未成功触球。
- 当前最短 curriculum 要求连续完成 2 次触球；2-touch 短序列 missed 后立即结束，更长序列清空连续成功 streak 后继续。
- curriculum 接触数为 `(2, 4, 8, full, full)`，晋级阈值为 sequence completion `0.60`，并要求低 fall rate。

## 8. 明确关闭的球 Reward

S2 不使用以下通用或历史球 shaping：

- pelvis/球关系：`dribbling_face_ball`、`dribbling_dynamic_proximity`、`dribbling_stall_no_touch_penalty`；
- 脚球通用 shaping：`dribbling_approach_foot_ball`、`dribbling_legal_foot_touch`；
- 球速度、推进、追球：`dribbling_velocity_tracking`、`dribbling_ball_speed_excess`、`dribbling_ball_coast_penalty`、`dribbling_ball_forward_progress`、`dribbling_chase_ball`；
- 夹球、连续接触和弹跳：`dribbling_ball_trapped_penalty`、`dribbling_sustained_contact_penalty`、`dribbling_ball_bounce_penalty`、`dribbling_orbiting_penalty`；
- 旧接触 shaping：`dribbling_rapid_retouch_penalty`、`dribbling_micro_contact_filter`、`dribbling_undesired_contact_penalty`；
- 旧 CG 项：`dribbling_cg_flow_release`、`dribbling_cg_flow_progress`、`dribbling_cg_contact_consistency`、`dribbling_cg_premature_contact`、`dribbling_cg_foot_consistency`；
- 普通脚 tracking：`motion_foot_pos`、`dribbling_gait_foot_tracking`、`foot_distance`，统一由 `s2_windowed_foot_tracking` 代替。

## 9. 一句话总结

6.1 用 reference 时刻、指定脚和 inside/outside 脚面监督每次独立物理触球：接触前由窗口化脚跟踪和 proximity 提供局部梯度，触球成功立即支付 `+4.5`；触球之后到下一事件之间没有专门评价球是否变得更可达。
