# Footmimic MDP 05：右脚单侧运球与真实机器人接触事件

本文是 `MDP_SUMMARY_04.md` 的后续版本，只覆盖当前保留的两个环境：

- Stage 1：`Tracking-CG-G1-Motion-RNN-mimic`
- Stage 2：`Tracking-CG-G1-Dribbling-RNN-control`

05 版不增加 reward，不调整 reward weight、observation 维度、action 维度或
termination 阈值。本次只修正接触信号、CG 接触评分和合法触球脚的语义。

## 1. 本次结论

新 diagnostic 表明，04 版的球净水平接触力不能作为“机器人是否碰球”的事实信号：

- 全局净力接触率约为 `13.44%`；原始逐-link 机器人接触率约为 `5.11%`。
- 242 个全局接触 sample 中只有 91 个与同帧逐-link 接触重合。
- 151 个“仅全局接触”sample 中，约 99 个不能由此前两帧的逐-link 接触解释。
- 这些 sample 的平均球速仍约为 `1.586 m/s`，但平均竖直速度接近零，符合滚动球的地面摩擦信号。

因此，04 版把球—地面摩擦误认成了机器人触球，并让 forward-progress、orbit、coast、
proximity 和 no-contact termination 的门控长期保持错误状态。

同一 diagnostic 还显示，CG 标注接触帧只占约 `23.1%`。旧的逐帧
`ref_contact == sim_contact` 在策略从不触球时也能取得约 `76.9%` 的正确率；实际旧项均值
约为 `0.753`。它主要奖励“双方都没接触”，而不是奖励完成触球。

## 2. 从 04 到 05 的变化

| 项目 | 04 版 | 05 版 |
|---|---|---|
| motion 数据 | 用户提供的右脚 `master-v2` | 不变；不镜像、不加入其他踢球数据 |
| 合法触球脚 | 默认左右 ankle 均合法 | 仅 `right_ankle_roll_link` 合法 |
| reward / termination 接触事实 | 球的净 XY 接触力 | 过滤后的机器人逐-link 接触，当前帧加后续 2 步保持 |
| safety 接触事实 | 原始逐-link | 不变；不使用保持信号 |
| 球净接触力 | 参与决策 | 仅用于 diagnostics 遥测 |
| CG consistency | 每帧 contact/no-contact 相等得 1 | 接触窗口事件得分；不奖励 true negative |
| reward 数量与权重 | Stage 2 为 27 项 | 完全不变 |
| input / action | actor 160、critic 292、action 29 | 完全不变 |

左脚仍参与双脚 gait/mimic tracking，因为正常走路需要两条腿；但左脚碰球现在属于
`dribbling_undesired_contact_penalty`。这不会要求额外左脚数据。

仓库内的 `master-v2` 已核验为单个 331 帧文件：`kick_leg=right`，90 个 CG contact
标注的 foot id 全部为 `1`（右脚）。训练脚本默认让 Stage 1 继续使用
`motions/master-modified` 做通用 mimic 预训练，而 Stage 2 只使用
`motions/master-v2`；两阶段共享网络输入接口，但不再错误地共享一个 motion path。

## 3. 三种接触量及用途

### 3.1 统一的机器人接触事实

```text
raw_link_contact(t) = max_link ||F_ball<-robot_link,xy(t)|| > 1.0 N

robot_contact(t) = raw_link_contact(t)
                OR raw_link_contact(t-1)
                OR raw_link_contact(t-2)
```

也就是一次真实逐-link 接触在当前 control step 以及之后 2 个 step 内有效，总持续最多
3 个 control step（约 `0.06 s`）。结果按 episode step 缓存，因此 reward、termination、
evaluation 和 play diagnostics 在同一步不会各自推进一次保持计数。

使用 `robot_contact` 的逻辑：

- `dribbling_ball_forward_progress` 的 recent-contact gate
- `dribbling_dynamic_proximity` 的 no-contact damping
- `dribbling_ball_coast_penalty` 的 release grace
- `dribbling_orbiting_penalty` 的 weak-contact gate
- `dribbling_gait_foot_tracking` 与 `dribbling_chase_ball` 的 no-contact gate
- `dribbling_cg_contact_consistency` 的窗口命中
- `dribbling_no_contact` termination
- play/eval 的统一接触率

### 3.2 原始逐-link 安全信号

```text
raw_link_force = max_link ||F_ball<-robot_link,xy||
```

以下安全项必须保留瞬时信号，不能被 2-step hold 延长：

- `dribbling_sustained_contact_penalty`：20-step contact-duty EMA
- `dribbling_ball_bounce_penalty`：只惩罚真实接触当帧的竖直弹跳
- `dribbling_legal_foot_touch`
- `dribbling_rapid_retouch_penalty`
- `dribbling_micro_contact_filter`
- `dribbling_undesired_contact_penalty`
- CG 窗口外的 premature contact
- diagnostics 的实际 contact body 与 wrong-foot 判断

### 3.3 球净接触力遥测

```text
ball_net_contact_force = ||F_ball_net,xy||
ball_net_contact = ball_net_contact_force > 1.0 N
```

它包含机器人、地面及其他接触的合力，并允许不同接触相互抵消。因此只保存在 diagnostic
用于分析，不再决定 reward 或 termination。

如果旧 IsaacLab runtime 没有 `force_matrix_w`，逐-link helper 仍会回退到净力以维持可运行性；
此时 `ball_contact_filter_available=false`，结果不能视为可靠的机器人接触实验。

## 4. CG 接触窗口事件评分

原始 CG contact 标注先在 motion 时间轴前后各扩展 2 帧：

```text
cg_window(t) = any(cg_contact_ref[t-2 : t+2])
```

每个连续窗口独立记录是否已命中：

```text
窗口内、尚未触球：                  0
窗口内、首次真实机器人触球及其后：   +1
窗口外、发生原始逐-link 触球：       -1
窗口外、没有触球：                   0
```

窗口结束、episode reset 或 motion phase 跳变时会清空命中状态。这样，策略不能再靠大量
no-contact/no-contact 帧获得正奖励；它必须在右脚参考接触窗口附近真正碰到球。

配置 weight 保持 `+1.0`，每 control step 的实际系数仍为 `+0.020`。没有新增 reward 项。

## 5. 当前 Stage 2 reward（27 项）

`control_dt = 0.02 s`，表中“每步系数”等于配置 weight 乘以 `0.02`。

| Reward | Weight | 每步系数 | 05 版用途 |
|---|---:|---:|---|
| `motion_anchor_lin_vel` | +5.0 | +0.100 | 有效 locomotion 线速度跟踪 |
| `motion_foot_pos` | +0.55 | +0.011 | 双脚 demo 相对位置 |
| `motion_body_pos` | +0.45 | +0.009 | 主要 body demo 风格 |
| `action_rate_l2` | -0.1 | -0.002 | effective action 平滑 |
| `joint_limit` | -10.0 | -0.200 | soft joint limit |
| `motion_anchor_ang_vel` | +1.0 | +0.020 | heading-error yaw-rate target |
| `locomotion_heading_tracking` | +1.0 | +0.020 | 有效 heading 跟踪 |
| `dribbling_pelvis_quat_tracking` | +2.0 | +0.040 | pelvis demo 姿态 |
| `dribbling_gait_foot_tracking` | +1.4 | +0.028 | 无真实机器人接触时保持 demo 步态 |
| `foot_distance` | +0.35 | +0.007 | 双脚最小间距 |
| `upper_body_reference_overflow` | -0.05 | -0.001 | 上肢 reference envelope |
| `dribbling_ball_forward_progress` | +7.5 | +0.150 | 真实接触后沿 command 推进；阈值统一为 1 N |
| `dribbling_dynamic_proximity` | +3.0 | +0.060 | 真实 no-contact 时的前方走廊门控 |
| `dribbling_ball_too_close_penalty` | -8.0 | -0.160 | 球过近的连续几何惩罚 |
| `dribbling_chase_ball` | +2.0 | +0.040 | 无真实接触且球在前方时追球 |
| `dribbling_face_ball` | +1.0 | +0.020 | pelvis 面向 command 且球在前方 |
| `dribbling_ball_speed_excess` | -2.5 | -0.050 | 球速超过 1.35 m/s 后惩罚 |
| `dribbling_ball_coast_penalty` | -2.2 | -0.044 | 真实触球后保留 8-step release grace |
| `dribbling_orbiting_penalty` | -6.0 | -0.120 | 无真实接触时惩罚绕球横移；公式和权重未改 |
| `dribbling_cg_foot_ball_distance` | +3.5 | +0.070 | CG 右脚—球 XY 距离 |
| `dribbling_cg_contact_consistency` | +1.0 | +0.020 | CG 窗口事件评分 |
| `dribbling_legal_foot_touch` | +5.5 | +0.110 | 仅奖励右 ankle 的新 gentle touch |
| `dribbling_rapid_retouch_penalty` | -6.0 | -0.120 | 合法触球间隔小于 26 steps |
| `dribbling_sustained_contact_penalty` | -6.0 | -0.120 | 原始逐-link 20-step duty EMA |
| `dribbling_micro_contact_filter` | -4.0 | -0.080 | 限制右 ankle 过强触球 |
| `dribbling_ball_bounce_penalty` | -3.0 | -0.060 | 原始接触当帧 `abs(vz)>0.32 m/s` |
| `dribbling_undesired_contact_penalty` | -12.0 | -0.240 | 右 ankle 以外的 link（包括左脚）触球 |

这些 weight 与 04 完全一致。本次没有调整 `orbiting_penalty`、`ball_too_close_penalty`
或其他 reward 强度，避免同时改变信号定义和奖励配比。

Stage 1 的 11 项 reward 和 4 项 termination 完全不变，详表见 `MDP_SUMMARY_04.md`。

## 6. Stage 2 termination

| Termination | 条件 | 05 版变化 |
|---|---|---|
| `time_out` | 10 s | 无 |
| `ball_lost` | grace 50 后距离 `>1.0 m` 或速度差 `>2.0 m/s` | 无 |
| `dribbling_no_contact` | grace 50 后无接触计数达到 50 | 改用带 2-step hold 的真实机器人接触 |
| `locomotion_manual_sequence_end` | manual sequence 结束且 `reset_on_end` | 无 |

command-change recovery、proximity recovery 及其阈值均未修改。

## 7. Input 与 action

05 版没有改变网络接口：

- actor：160 维
- critic：292 维
- action：29 维 joint action
- 球坐标：仅 `anchor_ball_polar`
- locomotion command：仅 `motion_locomotion_polar_cmd`

完整有序字段见 `MDP_SUMMARY_04.md`。因此 04 版的 160/292 checkpoint 结构保持兼容；原始
v5.0 的 172/304 输入 checkpoint 仍不能被当作同构输入层直接加载。

## 8. Diagnostics

reward 的实际 runtime 权重仍会保存：

- `reward_term_names`
- `reward_term_weights`
- `reward_term_step_weights`
- `reward_step_dt`
- `reward_terms`
- `step_reward`

接触字段的 05 版定义：

| 字段 | 含义 |
|---|---|
| `ball_contact` | reward/termination 使用的逐-link + 2-step hold 接触事实 |
| `ball_link_contact` | 原始逐-link 接触事实 |
| `ball_contact_steps_since_link` | 距最近原始逐-link 接触的 step 数；`3` 表示已超过保持窗口 |
| `ball_max_link_contact_force` | 最大机器人 link 水平接触力 |
| `ball_net_contact` | 球净水平接触力是否超过 1 N，仅遥测 |
| `ball_contact_force` | 球净水平接触力数值，仅遥测 |
| `cg_contact_window_active` | 当前是否处于扩展后的 CG 接触窗口 |
| `cg_contact_window_hit` | 当前窗口是否已经发生真实接触 |
| `cg_contact_event_score` | CG reward 的未加权原始值 |
| `cg_premature_contact` | 窗口外原始逐-link 触球 |
| `cg_missing_contact` | 当前窗口截至该 sample 尚未命中 |

play 结束摘要会分别输出 `contact_rate`、`raw_link_contact_rate` 和
`net_contact_rate`，并在窗口结束 sample 统计 `cg_window_hit`。

## 9. 明确删除或拒绝的内容

| 内容 | 原因 |
|---|---|
| 左脚/双脚合法触球模式分支 | 当前目标和数据都只有右脚；保留会允许策略用左脚规避约束 |
| 净力作为机器人接触 gate | diagnostic 已证明它大量包含地面摩擦 |
| CG 逐帧相等评分 | 类别严重不平衡，会奖励从不触球 |
| 左右镜像或额外踢球 motion | 用户只要求右脚，不扩充数据 |
| 新 reward 或临时调 weight | 先隔离已证实的信号错误，避免无依据叠加 shaping |

本次预期是恢复 04 在直线段已经接近 v5 的表现，同时修复转向段中“滚动球被持续视为接触”
和“CG 不触球也得分”两类系统性错误。是否达到最优基线仍需用同一 checkpoint 初始化、同一
command sequence 和同一训练步数生成新的 diagnostic 后确认。
