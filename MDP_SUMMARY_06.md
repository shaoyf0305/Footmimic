# Footmimic MDP 06：恢复 v5 task 主导性的 possession/recovery 更新

## 1. 更新目标

`diagnostic_20260816_142647.npz` 表明，当前策略在首次连续直线 250 steps 已达到
v5 水平，但一次漏触球后容易进入连续 reset。实际 reward 分组显示：

| 实际贡献 / control step | v5 历史诊断 | 05 诊断 |
|---|---:|---:|
| reference / CG style 正贡献 | 约 `+0.1011` | `+0.1014` |
| 球任务正贡献 | `+0.1504` | `+0.0794` |
| 球任务 penalty（共同项） | `-0.0560` | `-0.0744` |
| 新 too-close penalty | 无 | `-0.0225` |
| task 净贡献（含 locomotion command） | 至少 `+0.1691` | `+0.0473` |

问题不是配置 weight 普遍变小，而是真实逐-link 接触把离散触球之间的大量 task gate
关闭，同时过早打开 coast/orbit penalty。06 不恢复受地面摩擦污染的球净力接触，也不新增
reward；它用一个明确的有限 possession 状态恢复 v5 的有效任务覆盖。

## 2. 保持不变的接口

- Stage 1：input、reward、termination 全部不变。
- Stage 2：仍为 160-D actor input、292-D critic input、29-D action。
- Stage 2 reward 数量仍为 27，所有 reward weight 完全不变。
- 球坐标仍只有 `anchor_ball_polar`，locomotion command 仍只有 polar input。
- 只有右 ankle 是合法球接触；左脚、膝、腕等仍由逐-link penalty 处理。
- manual sequence 在普通失败 reset 后继续当前 speed/heading 段，不强制回到第一段。

## 3. 共享 possession 状态

共享控球状态只由过滤后的 `right_ankle_roll_link` 逐-link 水平接触力刷新：阈值仍为 `1 N`，并保留 2-step 传感器保持。
在此基础上增加共享状态：

```text
真实右脚踝逐-link 接触 -> steps_since_contact = 0
每个 control step             -> steps_since_contact += 1
steps_since_contact <= 30     -> possession = true
episode reset                 -> possession = false
```

缓存保存 `last_step`，因此 RewardManager 在同一步依次计算多个 reward 时只推进一次计数。
30 steps 对应约 `0.6 s`，覆盖当前 master-v2 中约 22-step 的平均触球间隔，同时仍是有限状态。

## 4. Reward 门控变化

所有 weight 保持 05 数值，仅调整以下已有 reward 的触发语义：

| Reward | Weight | 06 行为 |
|---|---:|---|
| `dribbling_ball_forward_progress` | `+7.5` | recent-contact window 从 8 增至 30 steps。 |
| `dribbling_dynamic_proximity` | `+3.0` | possession 内不应用 `0.12` no-contact 衰减；真正失去 possession 后才衰减。 |
| `dribbling_ball_coast_penalty` | `-2.2` | 30-step 触球释放期内关闭。 |
| `dribbling_orbiting_penalty` | `-6.0` | possession 内关闭；超过 30 steps 未重新触球且确实绕球才惩罚。 |
| `dribbling_gait_foot_tracking` | `+1.4` | possession 内关闭，避免 reference gait 在两次合法触球之间压过控球动作。 |

`chase_ball` 保持原公式，用于球领先时追赶；legal touch、rapid retouch、sustained contact、
micro contact、bounce 和 undesired contact 仍直接使用真实逐-link 接触，不使用 possession
伪造新触球。

## 5. Too-close 几何修复

05 按 command-frame 的前向投影判断 too-close。转向时，位于机器人侧面且实际不近的球也可能
因为前向投影很小而得到满额 penalty。06 改为 pelvis--ball 的实际 XY 距离：

```text
distance_xy >= 0.28 m -> 0
distance_xy <= 0.14 m -> 1
中间线性插值
```

权重仍为 `-8.0`。这继续惩罚真实近身夹球，但不再惩罚正常转向形成的侧向几何。

## 6. No-contact recovery

原 command-change recovery 保持不变。额外启用已有的有限 proximity recovery：

| 参数 | 数值 |
|---|---:|
| 最大恢复预算 | 30 steps |
| 最大球距 | `0.65 m` |
| 最大球/机器人相对速度 | `0.8 m/s` |
| 恢复期间 no-contact 计数增量 | `0.25` |

只有 ball-lost 尚未成立、球仍近且相对速度可恢复时才减慢计数。预算耗尽或球真正离开后，仍按
原 `grace=50`、`max_steps_without_contact=50` 终止。

## 7. Playback reset

`play_multi.py` 在 `env.step()` 返回 `done` 后显式调用 recurrent policy 的 `reset(dones)`。
这只清空新 episode 的 LSTM hidden state，不修改 manual command plan、segment index 或全局
sequence 计时。

## 8. Diagnostics

除 05 的 reward runtime weights、逐项贡献和接触字段外，06 新增：

- `possession_active`
- `possession_steps_since_contact`
- `ball_too_close_distance_xy`

终端摘要的 `too_close_rate` 改为实际 penalty 大于零的比例，并新增 `possession_rate`。

## 9. 验收目标

同一 Stage-1 初始化、master-v2 数据和测试 command 下：

| 指标 | 05 诊断 | 06 目标 |
|---|---:|---:|
| task net / reference | `0.47` | `>=1.3` |
| 球沿 command 前进速度 | `0.95 m/s` | `>=1.6 m/s` |
| pelvis XY 速度 | `1.23 m/s` | `>=1.6 m/s` |
| 平均绝对 heading error | `0.276 rad` | `<=0.12 rad` |
| 1800-step termination | 13 | `<=4` |
| too-close rate | `20.2%`（旧投影定义） | `<=10%`（真实 XY 定义） |

这些目标用于判断是否恢复到 v5 行为水平；总 reward 只有在使用相同 06 定义时才可直接比较。
