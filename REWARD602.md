# S2 Reward 6.0.2（仓库实施版）

本文记录在仓库 `6.1` 基础上实施的 S2 6.0.2 reward。目标只有两个：

```text
R_total = R_reference_tracking + R_ball_task + R_regularization/safety
```

Ball 就是 task，不作为低权重的附属 shaping。正向 reference tracking 与正向 ball task 的名义权重保持接近。

- Gym task：`Tracking-CG-G1-Dribbling-RNN-unified-s2-local-reference`
- 球完全由物理仿真驱动
- 不读取 reference 球位置、reference 球速度或 reference 球轨迹
- Reward 总数：**20 项**，其中 reference/regularization/safety 13 项，S2 ball task 7 项
- S2 RewardManager 只保留这 20 项；所有继承但未使用的 reward 都从 manager 配置中删除，不以 `weight=0` 保留
- 表中是 `RewardTermCfg.weight`；单步汇总仍乘环境步长 `dt=0.02`

## 1. 相对 6.1 的改动

1. 新增 `s2_global_foot_ball_distance = +3.5`：全程评价实际目标脚与实际物理球的距离；
2. `s2_contact_proximity` 从 `+0.5` 调到 `+1.0`，并彻底删除 front gate；
3. `s2_valid_contact_bonus` 从 `+4.5` 调到 `+5.0`；
4. `s2_nonfoot_ball_contact` 从 `-1.0` 调到 `-5.0`；
5. `s2_wrong_foot_contact` 从 `-0.25` 调到 `-1.0`；
6. `s2_premature_contact` 从 `-0.25` 调到 `-0.5`；
7. 清理 S2 reward 配置，只暴露最终使用的 20 项。

基础 reference tracking、regularization、termination、curriculum 和 observation/action contract 不变。

## 2. 最终 Reward 总表

### 2.1 Reference tracking / regularization / safety（13 项）

| Reward term | 权重 | 作用 |
|---|---:|---|
| `motion_body_pos` | `+0.72` | 跟踪指定 body 的相对位置。 |
| `motion_body_ori` | `+1.0` | 跟踪 pelvis、torso 和上肢等 body 的 reference 朝向。 |
| `motion_body_lin_vel` | `+0.3` | 跟踪 yaw 对齐后的 reference body 线速度。 |
| `motion_body_ang_vel` | `+0.3` | 跟踪 yaw 对齐后的 reference body 角速度。 |
| `motion_anchor_lin_vel` | `+5.0` | 跟踪 reference pelvis-local `[vx, vy]`，`std=0.60`。 |
| `motion_anchor_ang_vel` | `+1.0` | 跟踪 reference pelvis yaw rate，`std=1.50`。 |
| `action_rate_l2` | `-0.1` | 惩罚实际执行动作的突变。 |
| `waist_action_rate_l2` | `-0.25` | 抑制 waist yaw/roll/pitch 抖动。 |
| `joint_limit` | `-10.0` | 惩罚超过 soft joint limits 的关节位置。 |
| `undesired_contacts` | `-0.1` | 惩罚普通环境中的非法接触。 |
| `pelvis_orientation` | `-2.5` | 抑制 pelvis 侧倾、后仰和失稳。 |
| `upper_body_reference_overflow` | `-0.05` | 惩罚超出上肢 reference 包络的 action target。 |
| `upper_body_manifold_nullspace` | `-0.02` | 惩罚会被上肢 PCA 投影丢弃的 action 分量。 |

其中正向 reference tracking 加上 windowed foot tracking 的名义权重为：

```text
0.72 + 1.0 + 0.3 + 0.3 + 5.0 + 1.0 + 1.0 = 9.32
```

### 2.2 S2 Ball task（7 项）

| Reward term | 权重 | 输出 | 作用 |
|---|---:|---|---|
| `s2_windowed_foot_tracking` | `+1.0` | 连续 `[0,1]` | 保留 reference 步态，并在目标触球附近释放指定脚；计入 reference tracking。 |
| `s2_global_foot_ball_distance` | `+3.5` | 连续 `[0,1]` | 全程鼓励实际目标脚靠近实际球，覆盖非接触阶段。 |
| `s2_contact_proximity` | `+1.0` | 连续 `[0,1]` | 在事件前和 contact window 内约束可达距离及 inside/outside 脚面。 |
| `s2_valid_contact_bonus` | `+5.0` | 每事件一次脉冲 | 奖励正确时刻、正确脚和正确脚面的新物理触球。 |
| `s2_nonfoot_ball_contact` | `-5.0` | 新接触脉冲 | 惩罚身体、膝或手等非脚 link 碰球。 |
| `s2_wrong_foot_contact` | `-1.0` | 新接触脉冲 | 惩罚 contact window 内错误脚触球。 |
| `s2_premature_contact` | `-0.5` | 每目标至多一次 | 惩罚目标 contact window 之前发生的脚触球。 |

```text
R_ball = 3.5 * global_foot_ball_distance
       + 1.0 * contact_proximity
       + 5.0 * valid_contact
       - 5.0 * nonfoot_ball_contact
       - 1.0 * wrong_foot_contact
       - 0.5 * premature_contact
```

Ball task 的正向名义权重为：

```text
3.5 + 1.0 + 5.0 = 9.5
```

因此正向 reference tracking `9.32` 与正向 ball task `9.5` 接近。实际训练仍应按各项 episode return 校准，而不能只比较配置权重。

## 3. 全程实际脚球距离

Reward 配置名：`s2_global_foot_ball_distance`  
实现函数：`dribbling_s2_global_foot_ball_distance_exp`

只使用实际仿真位置：

```text
p_ball_sim          # 实际物理球世界位置
p_foot_sim_selected # 实际目标脚 body 世界位置
d_sim = ||p_ball_sim - p_foot_sim_selected||_2
r_global = exp(-(d_sim / 0.40)^2)
```

参数：

```text
global_foot_ball_distance_std = 0.40  # m
```

目标脚选择：

- 当前 contact window 内选择当前事件标注脚；
- 其他时间选择下一事件标注脚；
- 当前 window 结束后切换到下一事件指定脚；
- 没有当前或后续有效事件时输出 `0`。

事件标签只决定左脚或右脚，不开关距离 reward。只要存在当前或下一目标，该 reward 在接触前、接触中和两次接触之间都生效，并且不要求接触传感器触发。

| 实际脚球距离 | `r_global` | 加权后（`×3.5`） |
|---:|---:|---:|
| `0.25 m` | `0.676` | `2.366` |
| `0.40 m` | `0.368` | `1.288` |
| `0.50 m` | `0.210` | `0.735` |
| `0.70 m` | `0.047` | `0.165` |

宽尺度 `0.40 m` 保证非接触阶段仍有梯度；`3.5` 延续成功版本脚球距离 reward 的量级，使 ball task 不被 reference tracking 淹没。

## 4. Contact Proximity：无 Front Gate

`s2_contact_proximity` 使用实际目标脚 yaw-local 坐标中的球偏移，但只计算物理可达性与标注脚面：

```text
physical_distance = ||ball_from_actual_foot||
reach_error = relu(physical_distance - 0.25)
side_error = relu(0.04 - signed_lateral)
target_distance = sqrt(reach_error^2 + side_error^2)
region_score = 1 / (1 + (target_distance / 0.12)^2)
contact_proximity = region_score * active * time_weight
```

这里没有任何 `x_foot` 前后判断、front gate、前向硬阈值或有符号前向分数。球在脚前方还是后方不会单独改变 proximity；reward 只回答：

1. 实际指定脚是否进入球的物理可达范围；
2. 球是否位于 reference 标注的 inside/outside 脚面一侧。

生效时间保持 6.1 行为：

- 下一事件前 `0.30 s` 逐渐开启，`time_weight` 从 `0.20` 增加到 `1.0`；
- contact window 内完整生效；
- 当前事件第一次有效触球后关闭该事件的 proximity。

全程距离负责大范围脚球接近，contact proximity 负责触球附近的脚面几何，两者职责不同。

## 5. 有效触球与错误接触

有效触球必须同时满足：

1. 位于当前 reference contact window；
2. 当前步出现新的物理接触上升沿；
3. 接触 body 是合法脚 link；
4. 接触脚与事件指定脚一致；
5. 球位于指定的 inside/outside 脚面一侧；
6. 当前事件之前尚未成功。

每个事件最多支付一次：

```text
5.0 * dt = 0.10
```

错误接触继续共享同一 contact event state：

- 非脚触球：`-5.0 * dt = -0.10`；
- 错误脚触球：`-1.0 * dt = -0.02`；
- 提前触球：`-0.5 * dt = -0.01`，每目标至多一次。

`missed_valid_contact`、`ball_lost` 和 S2 curriculum 保持 6.1 行为，确保只靠近球但不完成有效触球不能完成任务。

## 6. RewardManager 清理规则

S2 使用最终 reward 白名单。环境构造结束后，只允许本文表中的 20 项进入 RewardManager；其他从通用 dribbling/CG 基类继承的 reward 全部删除。

这样做有三个目的：

- 不让零权重 term 留在配置、日志和运行时 bookkeeping 中；
- 防止父配置未来改权重时意外重新启用旧目标；
- 让 diagnostic 中的 reward 名称与实际训练目标一一对应。

共享 reward 函数仍可供其他环境使用；清理只作用于 S2 的最终 RewardManager，不改变 S1、S3 或通用 dribbling task。

## 7. Telemetry 与验收

新增并导出：

```text
s2_global_foot_ball_distance
s2_global_foot_ball_score
s2_global_foot_ball_expected_foot
s2_global_foot_ball_active
```

必须满足：

1. 距离下一 contact window 超过 `0.30 s` 时，只要存在下一事件，全程距离 reward 仍生效；
2. 没有物理接触时，全程距离 reward 正常计算；
3. 使用实际物理球与正确的实际左/右脚位置，不使用 reference 球状态；
4. 同一目标脚下，距离越小，分数严格单调增大；
5. 当前 window 结束后正确切换到下一事件脚；
6. 没有有效当前/后续事件时输出 `0`；
7. reset、resample 和 curriculum 截断后不残留旧事件状态；
8. proximity 代码、参数、state 和 diagnostic 中不存在 front gate；
9. RewardManager 中恰好有 20 项，其中 S2 专用项恰好 7 项；
10. RewardManager 中不存在权重为零的占位 reward。

训练时分别汇总：

```text
reference_tracking_return
ball_task_positive_return
ball_task_penalty_return
ball_to_reference_positive_ratio
```

建议将实际 `ball_task_positive_return / reference_tracking_return` 维持在 `0.8～1.2`，并同时检查有效触球率、sequence completion、提前触球率和悬停贴球率。

## 8. 一句话总结

6.0.2 用 `+3.5` 的全程实际脚球距离补齐非接触阶段信号，用无 front gate 的局部 proximity 约束可达性和指定脚面，再用真实有效触球完成任务判定；reference tracking 与 ball task 获得接近的正向权重，同时 S2 RewardManager 只保留最终使用的 20 项。
