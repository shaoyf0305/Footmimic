# S1 / S2 / S3 Reward 总表（以当前代码为准）

本文按当前提交 `eac8b2d` 的**最终环境配置**整理，而不是按 reward 函数是否存在来整理。这里的“生效”指：

- reward term 最终没有被设成 `None`；
- 最终 `weight != 0`；
- 表中的权重是 `RewardTermCfg.weight`。正数是奖励，负数是惩罚；reward 函数通常先返回非负原始值，再由负权重把它变成惩罚。

当前三阶段训练脚本使用的任务是：

| 阶段 | Gym task id | 最终配置类 | 生效 reward 数 |
|---|---|---|---:|
| S1 | `Tracking-CG-G1-Motion-RNN-unified-s1-local-strict` | `G1FlatMotionCGPretrainUnifiedS1LocalStrictEnvCfg` | 11 |
| S2 | `Tracking-CG-G1-Dribbling-RNN-unified-s2-local-reference` | `G1FlatCGDribblingUnifiedS2LocalReferenceEnvCfg` | 20 |
| S3 | `Tracking-CG-G1-Dribbling-RNN-unified-s3-local-task` | `G1FlatCGDribblingUnifiedS3LocalTaskEnvCfg` | 30 |

> 当前只注册上面三个 local S1/S2/S3 task；旧 S1/S2/S3、历史 baseline 和四个 S2 ablation task 已删除。

## 1. 总体目标

- **S1：严格动作模仿。** 没有任何球任务 reward。策略跟踪完整示范的相对身体姿态、身体速度和 pelvis-local root twist，同时受动作平滑、关节限位、非法落地接触和上身动作流形约束。
- **S2：参考接触事件模仿。** 保留较软的动作/上身先验和参考 local twist；球只由物理推动。球监督被收缩为 5 个事件 reward：接近指定脚面区域、指定脚首次触球、正确内/外脚背、柔和触球力、错误触球。没有球速度、球推进或参考球轨迹 reward。
- **S3：通用任务盘带。** local twist 从任务采样，不再规定参考触球时刻、触球脚或内/外侧标签；通过控球距离、球推进、追球、合法 instep 触球以及各种反作弊惩罚学习可控盘带。动作模仿仍作为软风格先验存在。

记号：

- `clip(x,a,b)`：把 `x` 截断到 `[a,b]`；`relu(x)=max(x,0)`。
- `p_sim/p_ref`、`q_sim/q_ref`、`v_sim/v_ref`：仿真值和示范参考值。
- `σ`：配置中的 `std`。
- `a_eff`：经过上身参考包络、PCA 投影、限位和滤波后真正交给 action term 的归一化动作。
- pelvis-local 坐标只使用 pelvis yaw：`+X` 是当前身体朝前，`+Y` 是当前身体左侧。
- 除非另有说明，盘带 reward 的“球接触力”是球传感器净力的 **XY 模长**，这样不会把地面对球的主要 Z 向支撑力误判成机器人触球。

## 2. S1：严格 local motion pretrain

S1 的最终 reward 为：

| Reward term | 权重 | 原始值/主要参数 | 作用 |
|---|---:|---|---|
| `motion_body_pos` | `+1.0` | `exp(-mean_i(‖p_ref,i-p_sim,i‖²)/0.3²)`；覆盖 motion command 中全部 body | 跟踪完整示范的相对身体位置，是主要姿态模仿项。 |
| `motion_body_ori` | `+1.0` | `exp(-mean_i(angle(q_ref,i,q_sim,i)²)/0.4²)`；全部 body | 跟踪完整示范的身体朝向。 |
| `motion_body_lin_vel` | `+1.0` | `exp(-mean_i(‖v_ref,i-v_sim,i‖²)/1.0²)`；世界系、全部 body | 跟踪示范各 body 的线速度，保留原动作的动态节奏。 |
| `motion_body_ang_vel` | `+1.0` | `exp(-mean_i(‖ω_ref,i-ω_sim,i‖²)/3.14²)`；世界系、全部 body | 跟踪示范各 body 的角速度。 |
| `motion_anchor_lin_vel` | `+5.0` | `exp(-‖(vx,vy)_cmd-(vx,vy)_pelvis‖²/0.45²)`；两者均在当前 pelvis yaw-local 系 | 精确跟踪参考 root 的前向/侧向 local 速度；替代绝对世界路径跟踪。 |
| `motion_anchor_ang_vel` | `+2.0` | `exp(-(wz_cmd-wz_pelvis)²/0.80²)` | 跟踪参考 root yaw rate。 |
| `action_rate_l2` | `-0.1` | `clip(sum((a_eff(t)-a_eff(t-1))²),0,100)` | 抑制实际执行动作的突变和抖动。这里使用投影、限位、滤波后的有效动作，不是原始 policy 输出。 |
| `joint_limit` | `-10.0` | Isaac Lab `joint_pos_limits`；全部关节 | 强惩罚关节位置越过 soft joint limits，保证物理可行性。 |
| `undesired_contacts` | `-0.1` | Isaac Lab `undesired_contacts`；接触阈值 `1.0` | 惩罚脚踝和手腕以外 body 的地面/环境接触，减少摔倒、膝盖或身体撑地。 |
| `upper_body_reference_overflow` | `-0.05` | `mean(sqrt(1+(overflow/0.25)²)-1)` | 惩罚上身目标超出 `q_ref ± 0.25 rad` 参考包络的部分，防止 policy 长期依赖后级 clamp。 |
| `upper_body_manifold_nullspace` | `-0.02` | `sqrt(1+(residual/0.10)²)-1` | 惩罚会被 6 维上身 PCA 流形投影丢弃的动作分量，让 policy 不把输出浪费在 null space。 |

### S1 的关键结论

1. S1 的球虽然为了保持 163-D observation contract 而生成在参考首次触球位置，但**没有任何球 reward**。
2. `motion_global_anchor_pos` 和 `motion_global_anchor_ori` 最终权重都为 `0`。因此 S1 不要求复现不可部署的绝对世界路径和绝对 yaw，而是跟踪 local `[vx, vy, wz]`。
3. body pose/velocity 仍是 strict full-body tracking；“local”只改变 root twist 的监督方式，并没有把 S1 变成弱模仿。

## 3. S2 / S3 共同生效的基础 reward

S2 和 S3 都继承下面 13 项。两阶段的区别主要在 local twist 的 `std` 以及后续球 reward。

| Reward term | S2 权重 | S3 权重 | 原始值/主要参数 | 作用 |
|---|---:|---:|---|---|
| `motion_body_pos` | `+0.72` | `+0.72` | `exp(-mean_i(‖p_ref,i-p_sim,i‖²)/0.3²)`；pelvis、左右 hip/knee、torso、双肩/肘/腕 | 软化后的身体位置风格先验；允许为了物理控球偏离示范。脚踝由阶段自己的逻辑处理。 |
| `motion_body_ori` | `+1.0` | `+1.0` | `exp(-mean_i(angle²)/0.4²)`；pelvis、torso、双肩/肘/腕 | 保持骨盆和上身朝向/姿态；不直接约束腿部朝向。 |
| `motion_body_lin_vel` | `+0.3` | `+0.3` | `exp(-mean_i(‖v_ref_aligned-v_sim‖²)/1.0²)`；同一组上身 body | 轻量跟踪上身线速度。参考速度先按 anchor yaw 对齐，避免绝对世界方向冲突。 |
| `motion_body_ang_vel` | `+0.3` | `+0.3` | `exp(-mean_i(‖ω_ref_aligned-ω_sim‖²)/3.14²)`；同一组上身 body | 轻量跟踪上身角速度和动作节奏。 |
| `motion_anchor_lin_vel` | `+5.0` | `+5.0` | pelvis-local XY 速度指数跟踪；S2 `σ=0.60`，S3 `σ=0.65` | S2 跟踪示范 local `[vx,vy]`；S3 跟踪采样且在开头逐渐从 S2 reference blend 过来的任务 `[vx,vy]`。 |
| `motion_anchor_ang_vel` | `+1.0` | `+1.0` | yaw rate 指数跟踪；`σ=1.50` | 跟踪 local task/reference 的 `wz`。 |
| `action_rate_l2` | `-0.1` | `-0.1` | 与 S1 相同，使用 `a_eff` | 平滑真正执行的全身动作。 |
| `waist_action_rate_l2` | `-0.25` | `-0.25` | `clip(sum_waist((a(t)-a(t-1))²),0,100)` | 对 waist yaw/roll/pitch 额外加大动作变化惩罚，减少腰部抖动。它与全身 `action_rate_l2` 同时生效。 |
| `joint_limit` | `-10.0` | `-10.0` | 全部关节 | 防止关节越过 soft limits。 |
| `undesired_contacts` | `-0.1` | `-0.1` | 阈值 `1.0`；排除左右脚踝和左右手腕 | 惩罚普通环境接触中的膝、躯干等非允许 body 接触。不要和下面专门判断“机器人—球”的惩罚混淆。 |
| `pelvis_orientation` | `-2.5` | `-2.5` | pelvis 局部重力向量横向分量平方和：`gx²+gy²` | 强化直立，抑制后仰、侧倒和弓背式投机。 |
| `upper_body_reference_overflow` | `-0.05` | `-0.05` | 与 S1 相同 | 约束上身 action 目标在示范附近的安全包络。 |
| `upper_body_manifold_nullspace` | `-0.02` | `-0.02` | 与 S1 相同 | 减少会被上身 PCA 投影丢弃的无效 action。 |

## 4. S2：参考 contact-event reward

S2 在上面 13 个共同项之外，还生效 7 项，共 20 项。

### 4.1 参考姿态与脚轨迹

| Reward term | 权重 | 原始值/主要参数 | 作用 |
|---|---:|---|---|
| `dribbling_pelvis_quat_tracking` | `+2.0` | `exp(-angle(q_ref_pelvis,q_sim_pelvis)²/0.45²)` | 额外跟踪参考 pelvis quaternion，防止为了够球而产生明显错误的 pelvis 姿态。它与 `motion_body_ori` 和直立惩罚同时存在。 |
| `s2_windowed_foot_tracking` | `+1.0` | 双脚位置误差指数，`σ=0.30`；支撑脚权重始终为 `1`；触球脚在事件前 `0.30 s` 从 `1` 线性降到 `0.10`，事件后窗口内保持 `0.10` | 非触球阶段跟踪双脚 gait；接近触球时释放指定触球脚，让物理和球的位置决定最后落脚，同时保持支撑脚稳定。 |

### 4.2 共享的 S2 触球事件判定

后面 5 个 reward 共用一次缓存的 `dribbling_s2_contact_event_state()`，所以它们对“本步是否是新触球、哪只 body 触球、力多大”的判断完全一致。

当前主要条件：

- reference contact window 为事件中心前后各 `0.10 s`；接近 shaping 从事件前 `0.30 s` 开始。
- 指定脚必须正确：`require_expected_foot=True`。
- inside/outside instep 标签启用：`target_side_enabled=True`，脚面侧向 deadzone 为 `0.04 m`。
- 有效触球必须是 contact 的上升沿、处于 reference window、由指定脚踝完成，且冲击力 `<=150 N`。
- 球始终由物理仿真驱动；reward **不读取 reference 球位置或 reference 球速度**。
- 球传感器没有给出具体碰撞点；代码用离球心最近的候选 robot body 估计触球 body，并在所选脚的 yaw-local 坐标中用球心横向偏移近似 inside/outside surface。

| Reward term | 权重 | 原始值/主要参数 | 作用 |
|---|---:|---|---|
| `s2_contact_proximity` | `+1.0` | `1/(1+(d_target/0.12)²) × front_gate × time_weight` | 给指定脚接近球的连续梯度。`d_target=sqrt(reach_error²+side_error²)`：距离在 `0.25 m` 内没有 reach error，球进入正确内/外侧 `0.04 m` 外没有 side error。球在脚后方时，`front_gate` 在局部 `x=-0.05..0.05 m` 从 0 线性升到 1。事件前 0.30 s 的 `time_weight` 从 0.20 升到 1；该事件一旦成功，后续 dense reward 被关掉。 |
| `s2_new_touch` | `+10.0` | 每个事件最多一个 `1` 脉冲 | 当前事件窗口内，指定脚第一次以 `<=150 N` 接触球就奖励。这是 S2 最强的稀疏成功信号。注意它本身不要求 inside/outside 正确，脚面侧别由下一项单独加分。 |
| `s2_correct_side` | `+4.0` | 正确侧的新触球返回 `1` | 在 `s2_new_touch` 成立的同时，球位于 reference 标注的 inside/outside instep 一侧且越过 `0.04 m` deadzone 时追加奖励。 |
| `s2_touch_force_soft` | `-1.0` | `clip(relu(F-80)/(150-80),0,2)`；仅指定脚的新物理接触 | 从 `80 N` 开始连续惩罚冲击力，使触球在到达 `150 N` 硬失效上限之前就有“更轻”的学习梯度。 |
| `s2_undesired_ball_contact` | `-6.0` | 非脚踝 `1.0`；错误脚 `0.5`；事件前过早脚踝触球 `0.5` | 惩罚身体/膝/腕等非脚踝触球（实际贡献 `-6`），以及错误脚或过早触球（各实际贡献 `-3`）。错误 inside/outside 不在此项直接扣分，而是少拿 `s2_correct_side` 并降低 proximity。 |

### 4.3 S2 实际没有在优化什么

这是当前代码最容易被旧文档误导的地方。S2 的以下项全部已移除或关闭：

- 通用球 shaping：`dribbling_face_ball`、`dribbling_velocity_tracking`、`dribbling_dynamic_proximity`、`dribbling_stall_no_touch_penalty`、`dribbling_approach_foot_ball`；
- 球速度/推进/追球：`dribbling_ball_speed_excess`、`dribbling_ball_coast_penalty`、`dribbling_ball_forward_progress`、`dribbling_chase_ball`；
- 反夹球/反连续接触：`dribbling_ball_trapped_penalty`、`dribbling_sustained_contact_penalty`、`dribbling_ball_bounce_penalty`、`dribbling_orbiting_penalty`；
- 旧触球 reward：`dribbling_legal_foot_touch`、`dribbling_rapid_retouch_penalty`、`dribbling_micro_contact_filter`、`dribbling_undesired_contact_penalty`；
- 旧 CG reward：`dribbling_cg_flow_release`、`dribbling_cg_flow_progress`、`dribbling_cg_contact_consistency`、`dribbling_cg_premature_contact`、`dribbling_cg_foot_consistency`；
- 普通 `motion_foot_pos`、`dribbling_gait_foot_tracking`、`foot_distance`，由 `s2_windowed_foot_tracking` 统一替代；
- `motion_global_anchor_pos`、`motion_global_anchor_ori`、`forward_velocity`、`lateral_velocity_penalty`、`task_heading_alignment` 的最终权重为 `0`。

因此，当前 S2 的核心不是“让球持续按 reference 速度前进”，而是“在保留动作/local twist 先验的同时，按 reference 事件序列用指定脚和指定脚面侧完成真实物理触球”。

## 5. S3：local task dribbling reward

S3 使用第 3 节的 13 个共同项，并增加下列 17 项，共 30 项。S3 不再使用 reference contact time、reference contact foot、reference inside/outside side 或 reference ball trajectory。

### 5.1 步态与球相对位置

| Reward term | 权重 | 原始值/主要参数 | 作用 |
|---|---:|---|---|
| `foot_distance` | `+0.35` | 两脚距离 `>=0.24 m` 时为 1；小于阈值时按 `exp(-((d/0.24-1)²)/0.5²)` 衰减 | 防止双脚交叉或重叠，给物理盘带留出稳定支撑面。 |
| `dribbling_face_ball` | `+2.5` | 当前 pelvis-local 中 `clip(x_ball/sqrt(x_ball²+y_ball²),0,1)`；距离 `<0.05 m` 时为 1 | 鼓励球位于当前 pelvis 正前方，而不是身体侧面或后方；不依赖世界 yaw。 |
| `dribbling_dynamic_proximity` | `+3.0` | `exp(-(forward_error²+lateral²)/0.15²)`；安全前向区间 `[0.28,0.72] m` | 把球保持在 pelvis-local 的前方走廊。还乘 `clip(pelvis_xy_speed/0.35,0,1)`，防止静止刷分；球在走廊且没有 `>1 N` 接触时再乘 `0.12`。走廊判定的横向上限为 `0.14 m`。 |
| `dribbling_stall_no_touch_penalty` | `-5.5` | 球距 pelvis `<=0.52 m`、pelvis 速度 `<=0.16 m/s`、球接触力 `<=1 N` 时返回 1 | 专门打掉“球停在身前，机器人也停住但不触球”的局部最优。 |
| `dribbling_gait_foot_tracking` | `+1.4` | 无球接触时，`exp(-mean(双脚位置误差²)/0.28²)` | 两次触球之间继续跟随 reference gait，避免一直保持单脚前伸的待踢姿态；一旦接触球，本项自动为 0，让 task 接管脚。 |

### 5.2 球速度、推进和追球

| Reward term | 权重 | 原始值/主要参数 | 作用 |
|---|---:|---|---|
| `dribbling_ball_forward_progress` | `+7.5` | 沿当前有效 local command 方向的球速；要求速度 `>=max(0.42,0.5×command_speed)`，在额外 `0.22 m/s` 内升到 1；横向/前向速度比在 `0..0.70` 把 gate 从 1 降到 0 | S3 的主要持续推进奖励。还要求 pelvis 速度 gate（`0.25 m/s` 达到 1）以及最近 `8` 步内有 `>0.5 N` 球接触，避免球自由滚动或侧踢刷推进。 |
| `dribbling_chase_ball` | `+2.0` | 无接触、球沿 command 方向领先 `>=0.28 m`、XY 距离 `<=1.05 m` 时，pelvis 前向速度从 `0.25` 到 `0.75 m/s` 线性给到 0..1 | 鼓励一次触球后主动追上滚远的球，形成“触球—追球—再触球”。 |
| `dribbling_ball_speed_excess` | `-2.5` | `clip(relu(norm(v_ball,xy)-1.35)/1.2,0,1)` | 抑制把盘带变成大脚开球；水平球速超过 `1.35 m/s` 后开始惩罚。 |
| `dribbling_ball_coast_penalty` | `-2.2` | 球距 pelvis `<=0.50 m` 且无 `>1 N` 接触时，`clip(relu(norm(v_ball,xy)-0.55)/0.40,0,1)` | 惩罚近身高速滑行却没有继续触球的球，减少依赖球自身惯性刷推进。远处追球阶段不罚。 |

### 5.3 合法触球和触球节奏

| Reward term | 权重 | 原始值/主要参数 | 作用 |
|---|---:|---|---|
| `dribbling_legal_foot_touch` | `+5.5` | 每个新的合法触球返回一个 `1` 脉冲；左右脚均可；力 `<=14 N`；surface=`instep` | 奖励轻柔的脚踝/脚背触球。S3 不要求指定左右脚，也不要求 reference 标注的 inside/outside；但球相对脚的横向偏移绝对值需至少 `0.018 m`，deadzone 内不算 instep。 |
| `dribbling_rapid_retouch_penalty` | `-6.0` | 新的脚踝轻触若距上次合法触球 `<26` env steps，则返回 1 | 防止每步快速点球或长时间黏球，鼓励离散触球节奏。这里用于判定的“轻触”上限是 `22 N`。 |
| `dribbling_micro_contact_filter` | `-4.0` | 脚踝触球力的 EMA：`ema=0.35F+0.65ema_prev`；`clip(relu(ema-22)/22,0,2)` | 连续惩罚过硬脚踝触球，避免只靠稀疏 legal-touch 阈值学习。 |
| `dribbling_undesired_contact_penalty` | `-12.0` | 非脚踝触球返回 `1`；脚踝落在 instep deadzone/错误 surface 返回 `0.25` | 强惩罚膝或腕触球（当前候选 body 是双脚踝、双膝、双腕；实际贡献 `-12`）；非 instep 脚踝触球较轻惩罚（实际贡献 `-3`）。 |
| `dribbling_sustained_contact_penalty` | `-6.0` | 球接触力 `>1 N` 连续超过 `3` 步后返回 1 | 防止持续挤压、夹住或拖着球走。 |
| `dribbling_ball_bounce_penalty` | `-3.0` | 接触球时 `abs(v_ball,z)>0.32 m/s` 返回 1 | 抑制夹球后向上弹起或把球踢飞。 |

补充：`dribbling_legal_foot_touch` 的奖励上限是 `14 N`，而 `dribbling_micro_contact_filter` 从 EMA 超过 `22 N` 才开始惩罚。因此 `14..22 N` 的正确 instep 触球不会得到 legal-touch 奖励，但也不会仅因力度立即触发 hard-contact EMA 惩罚。

### 5.4 反夹球、反绕球

| Reward term | 权重 | 原始值/主要参数 | 作用 |
|---|---:|---|---|
| `dribbling_ball_trapped_penalty` | `-8.0` | 球在 pelvis-local 前向距离 `<0.32 m`，或球高 `>0.20 m` 时返回 1 | 当前实现是“球太靠近/在身后/弹高”的代理惩罚，不是真正检测两脚夹球。S2 因会误伤参考姿态而关闭，S3 保留。 |
| `dribbling_orbiting_penalty` | `-6.0` | 球距 `<0.9 m` 且接触力 `<=1 N` 时，pelvis 绕球切向速度超过 `0.08 m/s` 后，在 `0.35 m/s` 范围线性升到 1 | 防止机器人不触球、只围着球绕圈或侧移刷其他 reward。 |

### S3 明确关闭的 reference / 重复项

- 所有 reference contact reward 均移除：`dribbling_cg_flow_release`、`dribbling_cg_flow_progress`、`dribbling_cg_contact_consistency`、`dribbling_cg_premature_contact`、`dribbling_cg_foot_consistency`、`dribbling_phase_graph_alignment`、`dribbling_support_ankle_roll`。
- `motion_foot_pos` 权重为 `0`；只保留无球接触时的 `dribbling_gait_foot_tracking`。
- `dribbling_approach_foot_ball` 权重为 `0`；S3 不用 dense 的“某只脚靠近球”奖励。
- `dribbling_velocity_tracking` 权重为 `0`；不直接奖励球速等于 pelvis 速度，推进由 `dribbling_ball_forward_progress` 负责。
- `dribbling_pelvis_quat_tracking` 权重为 `0`；S3 不要求复现 reference pelvis quaternion。
- `motion_global_anchor_pos`、`motion_global_anchor_ori`、`forward_velocity`、`lateral_velocity_penalty`、`task_heading_alignment` 权重为 `0`；local twist reward 是唯一 root command 跟踪目标。
- `target_point_proximity` 被删除；它属于旧 kick destination 任务，不属于连续盘带。

## 6. 三阶段 reward 迁移关系

| 能力 | S1 | S2 | S3 |
|---|---|---|---|
| 完整 strict body imitation | 强 | 软化为身体/上身先验 | 软化为身体/上身先验 |
| local `[vx,vy,wz]` | reference，较严格 | reference | sampled task；初期从 reference blend |
| 球 reward | 无 | 只监督 reference contact event | 通用物理盘带 shaping |
| 参考触球时间 | 无 | 强制 | 不使用 |
| 参考触球脚 | 无 | 强制 | 左右脚均可 |
| 参考 inside/outside | 无 | 正确侧追加奖励并进入 proximity | 不使用；任一 instep 侧均可 |
| 参考球位置/速度 | 无 | 不使用 | 不使用 |
| 球推进 | 无 | 不直接奖励 | 强奖励 |
| 防黏球/硬踢/绕球 | 无 | 由事件力和错误接触处理 | 多个专用惩罚 |

从 reward 设计上看，三阶段不是简单“逐项累加”：S2 用 reference event 把触球技能教出来；进入 S3 后，这套 reference contact supervision 被整体删除，再换成与 task command 一致的控球、推进、追球和合法触球目标。

## 7. 已删除的三阶段未使用 reward

删除旧 task 配置后，下面 28 个公开 reward 函数在 `source/` 和 `scripts/`
的 Python AST 中均为零引用，因此已一起删除：

- 旧 control / 通用实验项：`task_state_gated_reward`、`locomotion_heading_tracking_exp`、`motion_anchor_xy_speed_excess_penalty`、`motion_anchor_xy_speed_deficit_penalty`、`motion_anchor_ang_vel_tracking_exp`、`feet_contact_time`、`feet_slip_penalty`。
- trunk pitch 实验项：`trunk_pitch_reference_overflow_penalty`、`waist_pitch_reference_error_exp`、`trunk_relative_pitch_reference_error_exp`、`trunk_relative_pitch_rate_l2`、`trunk_pitch_effective_action_rate_l2`。
- 已放弃的 kick / 旧 CG 项：`target_point_contact`、`sideways_kick`、`ball_velocity_direction_alignment`、`ball_speed_reward`、`ball_z_speed_penalty_reward`、`early_collision_penalty`、`time_gated_contact`、`dynamic_ankle_masking_body_pos`。
- 旧 dribbling command / ablation 项：`dribbling_idle_stand_reward`、`dribbling_stop_settle_reward`、`dribbling_command_dynamic_proximity`、`dribbling_phase_graph_alignment`、`dribbling_support_ankle_roll_tracking_exp`、`dribbling_command_ball_trapped_penalty`、`dribbling_cg_foot_ball_distance_exp`、`dribbling_command_face_ball`。

同时删除了只服务这些 reward 的内部 helper：`locomotion_turn_relaxation`、
`_turn_stability_weight`、`_pitch_from_quat`、`_get_cg_phase` 和
`_locomotion_ang_vel_command_w`。

四个已删除的 S2 ablation 是累计消融环境，不是四个 reward：`motion` 只保留
动作/reference twist；`time` 加参考触球时机；`foot` 再加指定左右脚；`side`
最后加 inside/outside instep 侧别，等价于完整 S2 接触目标。

## 8. 代码来源

- 三个 task 的注册：[source/whole_body_tracking/soccer/tasks/tracking/config/g1/__init__.py](source/whole_body_tracking/soccer/tasks/tracking/config/g1/__init__.py)
- 三阶段最终配置和权重覆盖：[source/whole_body_tracking/soccer/tasks/tracking/config/g1/soccer_dribbling_env_cfg.py](source/whole_body_tracking/soccer/tasks/tracking/config/g1/soccer_dribbling_env_cfg.py)
- 基础 tracking rewards：[source/whole_body_tracking/soccer/tasks/tracking/tracking_env_cfg.py](source/whole_body_tracking/soccer/tasks/tracking/tracking_env_cfg.py)
- proximity / stage-1 配置：[source/whole_body_tracking/soccer/tasks/tracking/config/g1/soccer_flat_env_cfg.py](source/whole_body_tracking/soccer/tasks/tracking/config/g1/soccer_flat_env_cfg.py)
- 动作模仿、local twist、动作正则实现：[source/whole_body_tracking/soccer/tasks/tracking/mdp/rewards.py](source/whole_body_tracking/soccer/tasks/tracking/mdp/rewards.py)
- S2 event 与 S3 dribbling reward 实现：[source/whole_body_tracking/soccer/tasks/tracking/mdp/rewards_dribbling.py](source/whole_body_tracking/soccer/tasks/tracking/mdp/rewards_dribbling.py)
