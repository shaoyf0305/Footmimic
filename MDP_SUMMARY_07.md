# Footmimic MDP 07：仿真球速度闭环

## 1. 更新目标

06 已恢复右脚控球和 reset 鲁棒性，但 `diagnostic_20260816_224119.npz`
暴露了速度闭环缺失：`1.5 m/s` 命令下，完整转向段的球速达到
`3.45--3.73 m/s`。旧 progress 在约 `0.97 m/s` 后已经满分，旧
speed-excess 在 `2.55 m/s` 后又完全饱和，因此高速仍有净正收益，且
penalty 对更高速度没有梯度。

07 只解决仿真内速度控制。视觉检测、丢帧、置信度和部署 tracker 不在
本次范围内。

## 2. Input 变化

Stage 1 和 Stage 2 同时追加相同的三维仿真输入：

```text
anchor_ball_velocity_polar_cmd = [
    ball_xy_speed,
    cos(ball_velocity_heading - command_heading),
    sin(ball_velocity_heading - command_heading),
]
```

球近似静止时输出 `[0, 0, 0]`。输入直接来自仿真球的 XY 线速度；未来
如接入视觉 estimator，只需替换 observation 的数据源，不改变三维语义。

| 网络 | 06 | 07 |
|---|---:|---:|
| Actor | 160 | **163** |
| Critic | 292 | **295** |

原有球位置仍只有 `anchor_ball_polar`，locomotion command 仍只有
`motion_locomotion_polar_cmd`。新速度输入追加在这两个 polar term 后，原
160/292 维输入顺序不变。

Checkpoint loader 会为旧 checkpoint 的新三维输入追加零权重，并为
normalizer 追加 `mean=0、variance=1`；optimizer 状态在迁移时不恢复。

## 3. 新增球速度跟踪 reward

Stage 2 新增第 28 项 reward：

| Reward | Weight | 每步最大系数 |
|---|---:|---:|
| `dribbling_ball_velocity_tracking` | +2.5 | +0.050 |

它只在 DRIBBLE 状态且处于右脚 30-step possession 时生效。设：

```text
v_target  = effective locomotion command speed
v_forward = ball velocity projected on command heading
v_lateral = ball velocity projected on command lateral axis
tolerance = 0.15 + 0.10 * v_target
error     = relu(abs(v_forward - v_target) - tolerance)

reward = exp(-(error / 0.30)^2) * exp(-(v_lateral / 0.35)^2)
```

因此 `1.5 m/s` 命令的满分速度带约为 `1.20--1.80 m/s`。低速、反向和
高速都会降低 reward；侧向踢球也不能获得完整速度跟踪分数。

原 `dribbling_ball_forward_progress`、权重 `+7.5` 和 30-step gate 全部
保持不变。progress 继续提供“向前运动”的下界，新 reward 决定“接近哪个
速度”。

## 4. 已有超速 penalty 改为动态安全上限

`dribbling_ball_speed_excess` 数量和权重不变：

| Reward | Weight | 07 变化 |
|---|---:|---|
| `dribbling_ball_speed_excess` | -2.5 | 固定饱和 penalty 改为 command-relative Huber penalty |

公式为：

```text
speed_cap = max(0.35, v_target + 0.20)
x         = relu(ball_xy_speed - speed_cap) / 0.45

penalty = 0.5*x^2  (x <= 1)
          x - 0.5  (x > 1)
penalty <= 6
```

该 penalty 不使用 contact/possession gate，只要 DRIBBLE 中球仍高速运动就
持续生效。Huber 线性尾部避免高速梯度消失，`penalty <= 6` 将最坏每步
贡献限制为 `-0.30`，防止训练初期数值爆炸。

按 06 诊断轨迹离线反算：

| 区段 | 球速 | 新 tracking | 新 excess penalty |
|---|---:|---:|---:|
| 首段直线 | 1.56 m/s | +0.0299/step | -0.0100/step |
| 完整正转向 | 3.45 m/s | 约 0 | -0.1744/step |
| 完整负转向 | 3.73 m/s | 约 0 | -0.2093/step |

正常 v5 水平速度保留净正反馈，而 `3--4 m/s` 不再能用 progress 覆盖超速
代价。

## 5. 保持不变

- Stage 1 reward 和 termination 不变。
- Stage 2 原27项 reward 的配置 weight 不变；只新增一项速度跟踪 reward。
- 右脚逐-link 合法接触、30-step possession、proximity、coast、orbit、gait
  gate 全部不变。
- too-close、防夹球、bounce、undesired contact 和 contact consistency 不变。
- termination、reset、manual command sequence 和 LSTM reset 不变。
- Stage 2 command 训练范围仍为 `0.40--1.50 m/s`、heading `-0.75--0.75 rad`。

## 6. Diagnostics

07 新增：

- `ball_speed_target`
- `ball_speed_error`：沿 command 的有符号速度误差
- `ball_velocity_heading_error`
- `ball_speed_cap`
- `ball_velocity_tracking_reward`：未加权原始值
- `ball_speed_excess_penalty`：未加权原始值

已有 `reward_term_names`、runtime weights、逐项 reward、球速、possession 和
termination 字段继续保存。

## 7. 验收目标

使用 `0.5、1.0、1.5 m/s` 和 `0、+0.65、-0.65 rad` 组合测试：

| 指标 | 目标 |
|---|---:|
| 稳态球沿 command 速度 MAE | `<= 0.25 m/s` |
| `1.5 m/s` 命令的平均球速 | `1.25--1.80 m/s` |
| 球速超过 `2.5 m/s` 的比例 | `< 5%` |
| 平均绝对 heading error | `<= 0.12 rad` |
| 1800-step 实际失败 termination | `<= 2` |
| too-close rate | `<= 10%` |

速度合格的同时，右脚 contact、possession、防夹球和 reset 指标不得相对 06
退化。
