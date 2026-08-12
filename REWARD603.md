# S2 Reward 6.0.3（精简实施方案）

本文记录 S2 Reward 从 6.0.2 到 6.0.3 的改动内容和实施计划。表中权重为 `RewardTermCfg.weight`；RewardManager 单步汇总仍乘环境步长 `dt=0.02`。

**完整 Reward 变更表**

| 分类 | Reward term | 6.0.2 | 6.0.3 | 状态 | 作用 |
|---|---|---:|---:|---|---|
| Reference | `motion_body_pos` | `+0.72` | `+0.72` | 维持 | 跟踪指定 body 的相对位置。 |
| Reference | `motion_body_ori` | `+1.0` | `+1.0` | 维持 | 跟踪 pelvis、torso 和上肢 reference 朝向。 |
| Reference | `motion_body_lin_vel` | `+0.3` | `+0.3` | 维持 | 跟踪 yaw 对齐后的 reference body 线速度。 |
| Reference | `motion_body_ang_vel` | `+0.3` | `+0.3` | 维持 | 跟踪 yaw 对齐后的 reference body 角速度。 |
| Reference | `motion_anchor_lin_vel` | `+5.0` | `+5.0` | 维持 | 跟踪 pelvis-local `[vx, vy]`，`std=0.60`。 |
| Reference | `motion_anchor_ang_vel` | `+1.0` | `+1.0` | 维持 | 跟踪 reference pelvis yaw rate，`std=1.50`。 |
| Regularization | `action_rate_l2` | `-0.1` | `-0.1` | 维持 | 惩罚实际执行动作突变。 |
| Regularization | `waist_action_rate_l2` | `-0.25` | `-0.25` | 维持 | 抑制 waist yaw/roll/pitch 抖动。 |
| Safety | `joint_limit` | `-10.0` | `-10.0` | 维持 | 惩罚超过 soft joint limits。 |
| Safety | `undesired_contacts` | `-0.1` | `-0.1` | 维持 | 惩罚普通环境中的非法接触。 |
| Safety | `pelvis_orientation` | `-2.5` | `-2.5` | 维持 | 抑制 pelvis 侧倾、后仰和失稳。 |
| Regularization | `upper_body_reference_overflow` | `-0.05` | `-0.05` | 维持 | 惩罚超出上肢 reference 包络的 action target。 |
| Regularization | `upper_body_manifold_nullspace` | `-0.02` | `-0.02` | 维持 | 惩罚会被上肢 PCA 投影丢弃的 action 分量。 |
| S2 | `s2_windowed_foot_tracking` | `+1.0` | `+1.0` | 维持 | 保留 reference 步态，在触球附近释放目标脚。 |
| S2 | `s2_global_foot_ball_distance` | `+3.5` | — | 删除 | 删除可通过持续贴球刷取的全程绝对距离奖励。 |
| S2 | `s2_approach_progress` | — | `+2.0` | 新增 | 远距离阶段只奖励脚球距离改善。 |
| S2 | `s2_contact_proximity` | `+1.0` | `+1.0` | 维持 | 最后 `0.30 s` 和 contact window 内提供 side-aware 接近引导。 |
| S2 | `s2_valid_contact_bonus` | `+5.0` | — | 删除 | 拆成即时正确脚接触、脚面 style 和延迟 release。 |
| S2 | `s2_immediate_contact_bonus` | — | `+5.0` | 新增 | 奖励正确窗口、正确脚的新物理接触，不硬性要求脚面。 |
| S2 | `s2_next_touch_release` | — | `+12.0` | 新增 | 评价触球后方向、宽松速度带和脚球分离；首轮使用较保守权重，避免压过 imitation 与接近项。 |
| S2 | `s2_surface_style` | — | `+1.0` | 新增 | 低权重鼓励指定 inside/outside 脚面。 |
| S2 | `s2_nonfoot_ball_contact` | `-5.0` | `-5.0` | 维持 | 惩罚非脚 link 碰球。 |
| S2 | `s2_wrong_foot_contact` | `-1.0` | `-1.0` | 维持 | 惩罚错误脚触球。 |
| S2 | `s2_premature_contact` | `-0.5` | `-0.5` | 维持 | 惩罚目标窗口前的脚触球。 |
| 旧 Flow | `dribbling_cg_flow_release` | 不在 S2 白名单 | — | 删除 | 不接入旧的复杂 flow release/state。 |
| 旧 Flow | `dribbling_cg_flow_progress` | 不在 S2 白名单 | — | 删除 | 不启用逐段 flow progress。 |

6.0.3 计划启用 **22 项 reward**：13 项 reference/regularization/safety，加 9 项 S2。标记为删除的 term 必须从 S2 RewardManager 移除，不以 `weight=0` 保留。

## 一、本次改动内容

### 1. 改动目标与范围

1. `602n.npz` 中第一事件成功 `23/24`，第二事件成功 `0/23`，两触球完整序列为 `0`，已结束 episode 全部为 `missed_valid_contact`。
2. 第二触球长期失败的根因不是球直接生成在脚边：稳定阶段第一次触球前脚—球中心距离约 `0.24 m`，接触时约 `0.13 m`，回放中确实存在先伸腿再触球的过程。
3. 主要问题是 6.0.2 只判断“是否碰到球”，没有判断第一次触球是否把球送入第二次触球的可达状态。
4. 两个事件都是右脚，`s2_global_foot_ball_distance` 在事件切换后仍持续奖励右脚靠近球，无法区分正确释放、横向踢偏和持续贴球。
5. 本版只验证“双触球闭环”是否能学会，不同时解决 S3 任意方向泛化。

### 2. 接近 Reward 改为远近互补

1. 删除 `s2_global_foot_ball_distance`，避免静态贴球持续获得主要 ball reward。
2. 新增远距离 `s2_approach_progress`：

   ```text
   phi_t = exp(-(distance_foot_ball / 0.40)^2)

   r_approach = clamp((phi_t - phi_t-1) / dt,
                      -progress_max,
                      +progress_max)
   ```

3. 远距离阶段中，脚球距离改善为正、距离增大为负、保持不变约为 `0`。
4. 下一事件前最后 `0.30 s` 和 contact window 内关闭远距离 progress，切换到现有 `s2_contact_proximity`。
5. `s2_contact_proximity` 继续负责可接触距离及目标 inside/outside 一侧的近距离几何。
6. Immediate contact 后，两种 approach 都对当前事件关闭；8-step release 结束后才开始下一事件的 approach。

### 3. 即时接触与脚面 Style 分离

1. 删除原 `s2_valid_contact_bonus`。
2. 新增 `s2_immediate_contact_bonus`，触发条件仅为：

   - 当前位于 reference contact window；
   - 出现新的物理接触上升沿；
   - 接触 body 是合法脚 link；
   - 接触脚与当前事件指定脚一致。

3. Inside/outside 不再阻止即时接触和 release 学习。
4. 新增低权重 `s2_surface_style`：正确脚面得高分，错误脚面仍可以学习正确的出球方向。
5. 本版复用现有脚面判定，不同时重写 foot-yaw anti-exploit 几何；若低权重 style 仍导致明显扭脚，再在后续版本单独处理。

### 4. 实际触球时动态计算 Next-touch Release Target

1. Event table 对事件 `j` 只预存下一事件信息：

   ```text
   next_anchor_ref = reference ball position at touch event j+1
   next_event_frame = frame_j+1
   ```

2. Reference 触球锚点首轮直接使用 contact rising edge 的球位置；若单帧噪声明显，再改用 rising edge 前后极短窗口的中位数。
3. 不使用脚中心作为球锚点，也不跟踪两次事件之间的逐帧 reference 球轨迹。
4. 当事件 `j` 在仿真中实际触球时，使用实际状态重新计算：

   ```text
   actual_touch_frame = current reference frame at physical contact
   actual_ball_pos_w  = simulated ball position at physical contact

   next_anchor_w = transform_reference_anchor_to_current_episode(next_anchor_ref)
   delta_actual_w = next_anchor_w.xy - actual_ball_pos_w.xy
   remaining_time = (next_event_frame - actual_touch_frame) / fps

   target_direction_w = normalize(delta_actual_w)
   target_distance = norm(delta_actual_w)
   nominal_speed = target_distance / remaining_time
   ```

5. 动态计算会补偿实际触球早晚、触球前球的位置偏差以及当前 episode 的对齐偏差。
6. 例如 reference 事件为 frame 35 和 frame 67，若实际在 frame 30 触球，速度目标必须使用剩余 `37/50 = 0.74 s`，不能继续使用静态 reference 间隔 `32/50 = 0.64 s`。
7. `next_anchor_w` 复用 motion command 已有的 reference-to-current-episode 对齐变换。
8. `target_direction_w` 在实际触球时转换到世界坐标并锁存；release window 内不能随 pelvis yaw 再次旋转。
9. 若 `remaining_time <= 0`、`target_distance` 过小或下一锚点无效，则关闭本次 outgoing release 得分。

### 5. 简化的 8-step Release 评价

1. Immediate contact 时只锁存必要状态：

   ```text
   touch_event_id
   actual_touch_frame
   next_anchor_w
   target_direction_w
   target_distance
   remaining_time
   nominal_speed
   touch_ball_position
   streak_multiplier
   ```

2. 随后观察 8 个环境步（`0.16 s`）的实际球运动：

   ```text
   release_velocity = robust_average(actual_ball_velocity_after_touch)
   parallel_speed = dot(release_velocity, target_direction_w)
   lateral_speed  = abs(dot(release_velocity, perpendicular_axis))
   speed_ratio = parallel_speed / nominal_speed

   release_score = direction_score
                 * broad_speed_band_score
                 * separation_score
   ```

3. `direction_score` 奖励球从实际触球位置朝下一 reference 触球锚点运动。
4. 首轮速度满分/通过区间为：

   ```text
   0.65 <= parallel_speed / nominal_speed <= 1.35
   ```

   区间外平滑衰减，防止轻碰不够远或一脚踢飞，但不要求精确复现 reference 速度。
5. `separation_score` 要求触球后脚和球自然分离，避免持续夹球或高频连撞。
6. Release window 只判断初始出球质量；是否真正进入下一脚可达状态，仍由下一次 approach 和物理触球验证。
7. 本版不加入 force hard gate、精确球轨迹误差、progress segment 或旧的多阶段 flow 指标。

### 6. 连续高质量触球倍率

1. 只有通过 release 质量阈值的触球才增加 streak。
2. 当前触球使用之前已经完成的 streak：

   ```text
   multiplier(k_previous) = min(1.0 + 0.5 * k_previous, 2.5)
   ```

   | 当前触球 | 之前连续高质量触球数 | Release 倍率 |
   |---:|---:|---:|
   | 第 1 次 | 0 | `1.0` |
   | 第 2 次 | 1 | `1.5` |
   | 第 3 次 | 2 | `2.0` |
   | 第 4 次及以后 | >=3 | `2.5` |

3. 倍率只作用于 `s2_next_touch_release`，不放大 immediate contact、surface style 或 approach reward。
4. Missed contact、低质量 release、非法身体触球、ball lost、fall 或 reset 时清零 streak。

### 7. Level 0 加入 Second-touch Acquisition

1. Level 0 首轮训练分布：

   ```text
   75% 真实双触球 episode
   25% second-touch acquisition episode
   ```

2. 不加入 release-only single-touch，因为它不能证明连续运球能力。
3. 真实双触球继续从 frame 0 开始，两次触球之间完全由物理连续演化。
4. Acquisition 创建独立 episode，不在主序列中途 teleport：

   - 从第二事件前 `0.4～0.6 s` 的 reference robot state 开始；
   - 同时恢复同一 reference 时刻的球位置、线速度和角速度；
   - 当前 NPZ 没有独立 `ball_ang_vel_w`，首版用相邻球位置估计线速度，并按 `0.11 m` 半径的无滑动滚动关系推算角速度；
   - 初始脚—球距离来自真实 pre-contact 分布，初始时不得已经接触；
   - 在目标 contact window 前正常运行至少 20 个 observation step；
   - 完成第二事件的 approach、contact 和 release 后结束。

5. 在当前 `50 fps` reference 下，提前 `0.4～0.6 s` 对应 20～30 帧正常物理 rollout。Policy 持续观察球位置变化，LSTM 因而能在触球前估计球速。
6. 若扰动导致不足 20 个有效 observation step 就提前触球，拒绝该初始状态并重新采样；首轮位置/速度扰动保持较小。
7. 球不能直接生成在脚边；acquisition 必须保留“接近/伸腿再触球”的过程。
8. 本版只使用 reference pre-contact state 加小扰动，不实现成功 rollout-state buffer。
9. 只有真实双触球 episode 进入 Level 0 晋级统计；acquisition 成功率单独记录。
10. 真实双触球改善后，将 acquisition 比例从 `25%` 降至 `10%`，最后降至 `0%`，不实现复杂自适应退火器。

### 8. Input 与暂缓内容

1. Actor observation 保持现有 163-D，不增加 reference 球速度或 next-touch command：

   ```text
   154-D proprioception
   + 3-D actual ball relative position
   + 3-D fixed DRIBBLE state
   + 3-D body twist command [vx_local, vy_local, wz]
   = 163-D
   ```

2. 当前 RNN 通过连续球位置历史估计球速；simulator 球速度只用于 reward、critic 和 telemetry。
3. 6.0.3 暂不实现：

   - S3 body-twist-to-next-touch planner；
   - Actor next-touch command 或显式球速输入；
   - Counterfactual direction command；
   - 成功 rollout-state replay buffer；
   - Foot-yaw anti-exploit 重写；
   - 旧逐帧 `dribble_cg_flow_*` label/state/progress；
   - Hard-event replay 修改；
   - Level 1 之后的课程重构。

4. Level 0 通过后，后续课程暂时沿用：

   ```text
   2-touch -> 4-touch -> 8-touch -> full clip -> full clip + reset jitter
   ```

## 二、计划改动内容

### 1. RewardManager 与事件数据

1. 将 S2 RewardManager 白名单更新为表中 22 个启用项。
2. 从白名单删除 `s2_global_foot_ball_distance`、`s2_valid_contact_bonus` 和两个旧 flow reward。
3. 新增 `s2_approach_progress`、`s2_immediate_contact_bonus`、`s2_next_touch_release` 和 `s2_surface_style`。
4. Event table 为每个非最终事件预计算 `next_anchor_ref` 和 `next_event_frame`。
5. Source motion 最终事件没有下一触球点，不计算 outgoing release，只按到达接触处理。

### 2. Reward 状态与 Termination

1. 每个事件只保留最小状态：

   ```text
   APPROACH -> RELEASE_PENDING（8 steps） -> RELEASE_DONE
   ```

2. Immediate contact 同一事件最多支付一次。
3. `RELEASE_PENDING` 期间，`missed_valid_contact` 不能因 contact window 已结束而提前终止。
4. Release 质量失败不立即结束 episode，只将事件标为低质量并清零 streak。
5. 普通 missed contact、ball lost 和 fall 保持原终止逻辑。
6. Reset/resample 时清理 approach potential、release latch、streak 和 acquisition provenance。

### 3. 最小 Telemetry

计划新增或明确导出：

```text
s2_approach_progress
s2_contact_proximity
s2_immediate_contact
s2_release_pending
s2_release_actual_touch_frame
s2_release_next_anchor_w
s2_release_remaining_time
s2_release_target_distance
s2_release_target_direction
s2_release_nominal_speed
s2_release_parallel_speed
s2_release_lateral_speed
s2_release_speed_ratio
s2_release_separation_score
s2_release_score
s2_release_quality_pass
s2_quality_streak
s2_second_touch_conditional_success
s2_reset_mode
s2_acquisition_history_steps_before_contact
```

Reward return 至少分组为：

```text
reference_tracking_return
approach_return
immediate_contact_return
next_touch_release_return
surface_style_return
ball_penalty_return
```

### 4. Level 0 晋级指标

1. 只有无辅助的真实连续双触球 audit 参与晋级。
2. 主要记录：

   ```text
   真实双触球完成率
   P(第二次有效触球 | 第一次高质量 release)
   第一触球 release quality rate
   acquisition 第二触球成功率（仅诊断）
   fall rate
   ```

3. Hard-event replay 在 Level 0 中关闭，也不在本版修改。

### 5. 验收条件

1. RewardManager 中恰好存在 22 个启用项，不存在零权重占位 term。
2. 静态贴球不再持续获得主要 ball reward。
3. 远距离脚球接近产生 progress，最后 `0.30 s` 切换到 contact proximity。
4. Immediate contact 不依赖 inside/outside，surface 只影响低权重 style。
5. 横向或错误方向出球不能获得高 release score。
6. 提前或延迟触球时，方向、距离和 nominal speed 使用实际球位置及实际剩余时间重新计算。
7. `target_direction_w` 在触球时锁存，release window 内转动 pelvis 不改变目标方向。
8. `0.65 <= parallel_speed / nominal_speed <= 1.35` 为首轮速度通过区间，区间外平滑衰减。
9. Release pending 期间不会被 missed-contact 提前终止。
10. 第二次高质量触球的 release 倍率高于第一次，并在 `2.5` 封顶。
11. Acquisition 初始时无脚球接触，球位置和速度来自同一 pre-contact 时刻。
12. Acquisition 在目标触球前至少提供 20 个正常 observation step，常规范围为 20～30 帧。
13. Acquisition episode 不污染真实双触球晋级统计。
14. Actor observation 仍为现有 163-D，不包含 reference 球速度或新增方向 command。
15. Level 0 产生非零第二触球成功样本，并最终在无 acquisition 的真实双触球 audit 中改善。

### 6. 实施顺序

1. 更新 RewardManager 白名单，删除 global distance 和旧 valid-contact/flow 项。
2. 实现远距离 approach progress，并保留近距离 contact proximity。
3. 拆分 immediate contact 与 surface style。
4. 在 event table 保存下一 reference 触球锚点/帧。
5. 在实际触球当步动态计算并锁存世界坐标 release target。
6. 实现 8-step `s2_next_touch_release` 和 `0.65～1.35 × nominal_speed` 速度带。
7. 实现最小 `RELEASE_PENDING` 状态、streak multiplier 和 termination 对齐。
8. 更新最小 telemetry 和 return 分组。
9. 加入 25% second-touch acquisition，并强制触球前至少 20 个 observation step。
10. 使用固定 checkpoint 做诊断回放，再开始新训练。
