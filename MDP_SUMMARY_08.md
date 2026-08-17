# Footmimic MDP 08：球速闭环的动作质量修正

## 1. 为什么需要 08

07 成功压住了失控的高球速，但 `diagnostic_20260817_142716.npz` 表明它没有恢复到可用基线：

| 指标 | 06 | 07 | 判断 |
|---|---:|---:|---|
| 球沿 command 前向速度 | 2.348 m/s | 1.222 m/s | 超速被压住，但偏慢 |
| 球速 MAE | 1.172 m/s | 0.425 m/s | 明显改善 |
| `> 2.5 m/s` 比例 | 44.7% | 0.1% | 明显改善 |
| heading error | 0.088 rad | 0.163 rad | 退化 |
| possession | 91.6% | 78.7% | 退化 |
| 实际失败 termination | 2 | 4 | 退化 |
| 手臂参考误差 | 0.194 rad | 0.225 rad | 退化 |
| 腰部参考误差 | 0.382 rad | 0.554 rad | 严重退化 |

首段直线中，07 的总 reward 高于 06，但手臂误差增加约 25%、腰部误差增加约 21%、腰部 action step 增加约 61%。接触时的腰部 action step 从 06 的约 `0.87` 增至 07 的约 `1.66`。因此这不是单纯“没训练够”，而是 07 的目标函数允许策略用更差的动作换取更多球速收益。

根因有两个：

1. 07 保留 `dribbling_ball_forward_progress = +7.5`，又增加 `dribbling_ball_velocity_tracking = +2.5`，把球速相关正奖励预算从 `+7.5` 扩大为 `+10.0`。
2. tracking 直接跟踪逐控制步的碰撞后瞬时球速，容易诱导策略在每次触球附近做快速全身修正。

## 2. 08 的代码变更

### 2.1 保持原有球速正奖励预算

| Reward | 07 weight | 08 weight | 08 每步最大贡献（`dt=0.02 s`） |
|---|---:|---:|---:|
| `dribbling_ball_forward_progress` | +7.5 | **+5.0** | +0.100 |
| `dribbling_ball_velocity_tracking` | +2.5 | **+2.5** | +0.050 |
| 合计 | +10.0 | **+7.5** | **+0.150** |

08 不是删除前向进度：progress 仍负责避免静止和倒踢，velocity tracking 负责把速度收敛到 command。二者共享 06/5.0 水平的原始最大正奖励预算，动作质量 reward 不再被额外的 `+2.5` 任务收益盖过。

在 07 轨迹上做离线反算时，三个球速项（progress、tracking、excess）的平均每步合计贡献由 `0.11594` 降为 `0.07842`，接近 06 的约 `0.07263`；首段直线由 `0.16152` 降为 `0.11407`。该反算不代表新策略的最终表现，但确认 08 不再从目标函数层面放大球速收益。

### 2.2 tracking 改为 10-step 球速 EMA

控制频率为 50 Hz，因此 10 个控制步对应约 0.2 秒：

```text
alpha = 1 / 10

v_ema[t] = v_ball[t]                           # episode reset
v_ema[t] = 0.9 * v_ema[t-1] + 0.1 * v_ball[t]  # normal step
```

tracking 使用 `v_ema` 在 command 坐标系中的前向、侧向分量；状态按环境独立保存，并在 episode reset 时重置，不会把上一个 episode 的球速带入新场景。它每个 control step 只更新一次。

这项滤波只用于 reward。Actor/Critic 的三维球速输入仍是当前仿真真值，保留及时观测外界状态的能力；策略不再因为 reward 直接追逐单次碰撞尖峰。瞬时球速仍由超速 penalty 监管。

### 2.3 收紧目标速度满分带

08 的 tracking 为：

```text
v_target  = effective locomotion command speed
v_forward = EMA ball velocity projected on command heading
v_lateral = EMA ball velocity projected on command lateral axis
tolerance = 0.10 + 0.05 * v_target
error     = relu(abs(v_forward - v_target) - tolerance)

reward = exp(-(error / 0.30)^2) * exp(-(v_lateral / 0.35)^2)
```

在 `1.5 m/s` command 下，满分带由 07 的约 `1.20–1.80 m/s` 收紧为约 `1.325–1.675 m/s`，减少策略长期停在满分带下沿的空间。

### 2.4 瞬时超速安全项不变

`dribbling_ball_speed_excess = -2.5` 保持 07 的 command-relative Huber 形式，并继续读取未滤波的瞬时 XY 球速：

```text
speed_cap = max(0.35, v_target + 0.20)
x         = relu(ball_xy_speed - speed_cap) / 0.45

penalty = 0.5*x^2  (x <= 1)
          x - 0.5  (x > 1)
penalty <= 6
```

EMA tracking 管稳定速度，instantaneous excess 管危险峰值，两者职责不重复。

## 3. Input、reward、termination 的范围

- Stage 1/Stage 2 input 维度保持 Actor `163`、Critic `295`，本次不改网络接口。
- Stage 2 仍有 28 个 reward term，只修改上述两个正 reward 的分工和 tracking 参数。
- Stage 1 reward 不变。
- possession、右脚逐 link 合法接触、contact consistency、proximity、coast、orbit、gait、too-close、防夹球、bounce 和 undesired contact 全部不变。
- reset、termination、manual command sequence 和 playback LSTM reset 全部不变。
- command 训练范围仍为 speed `0.40–1.50 m/s`、heading `-0.75–0.75 rad`。

## 4. Diagnostics 增补

除 07 已有的实际 reward weights、逐项 reward、瞬时球速和 termination 信息外，08 新增：

- `ball_filtered_xy_speed`
- `ball_filtered_command_forward_speed`
- `ball_filtered_command_lateral_speed`
- `ball_filtered_speed_error`
- `ball_velocity_ema_window_steps`

终端摘要同时打印：

- `ball_speed_mae`：未滤波瞬时前向球速误差；
- `filtered_ball_speed_mae`：真正参与 tracking reward 的 EMA 前向球速误差。

这样可以区分“触球瞬间存在合理速度尖峰”和“稳定球速闭环本身没有跟上 command”。

## 5. 验收标准

08 的目标不是只让球慢，而是在恢复 5.0 动作质量和成功率的同时控制球速：

| 指标 | 目标 |
|---|---:|
| EMA 球速 MAE | `<= 0.25 m/s` |
| `1.5 m/s` command 的平均前向球速 | `1.25–1.75 m/s` |
| 球速超过 `2.5 m/s` 比例 | `< 5%` |
| 平均绝对 heading error | `<= 0.12 rad` |
| 1800-step 实际失败 termination | `<= 2` |
| possession | 不低于 06 的 90% |
| 手臂、腰部参考误差 | 至少回到 06 水平，继续向 5.0 靠近 |
| 接触附近腰部 action step | 明显低于 07，不再出现约 `1.66` 的异常峰值均值 |

如果球速指标合格但动作指标仍未恢复，不再继续加 reward；应优先检查当前 checkpoint 的训练阶段、mimic 初始化兼容性以及具体动作正则项的实际贡献。
