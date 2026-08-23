# Essay13 消融实现与运行说明

## 1. 实现原则

所有消融都位于同一个 Git commit，通过不同 Gym task 配置选择。`Ablation-Full` 是重新训练的完整方法控制组，其 MDP 配置与 Essay13 Stage-II 完整配置相同。除 No Stage-I 外，所有变体必须从同一个 Stage-I checkpoint 初始化，并使用相同的 motion bank、seed、环境数量和迭代预算。

训练入口会强制检查初始化约束。需要 Stage-I 的变体如果没有显式提供 run 和 checkpoint 会直接退出。No Stage-I 如果收到任何 resume 参数也会直接退出。

## 2. 已实现的变体

| 运行名 | 唯一变化 | Stage-I 初始化 |
|---|---|---:|
| `full` | 不移除任何因素，作为重新训练控制组 | 是 |
| `no_ball_velocity` | actor/critic 的三维球速度输入变为常量零，同时关闭直接球速度 tracking reward | 是 |
| `no_recovery` | gate 仍计算并记录，但 pelvis target 不再 blending，velocity controllability factor 恒为 1 | 是 |
| `no_stage1` | 完整 Stage-II MDP 从随机 actor、critic 和 LSTM 权重开始 | 否 |
| `no_dense_distance` | 只关闭 reference foot--ball distance reward | 是 |
| `no_touch_timing` | 只把 useful-touch 的 off-window scale 从 0.35 改为 1 | 是 |
| `no_interaction_reference` | 同时应用 no-dense 和 no-touch-timing | 是 |

扩展诊断变体也已实现：

| 运行名 | 唯一变化 |
|---|---|
| `no_ball_velocity_observation` | 只把 actor/critic 球速度输入变为常量零 |
| `no_ball_velocity_reward` | 只关闭直接球速度 tracking reward |
| `no_body_foot_reference` | 只关闭 Stage-II body-position 与 foot-position regularization |

没有实现 Task-only 组合变体。它会同时改变初始化、observation、reference reward 和 action semantics，不能回答单因素研究问题。如需保留，只能作为明确标记的附加组合比较另行设计。

## 3. 关键控制变量

### No explicit ball velocity

球速度 observation term 的名称、顺序和三维尺寸保持不变，仅把输出替换为常量零。这样 actor、critic、LSTM 和 checkpoint tensor shape 都与 Full 相同。球速过高惩罚仍作为公共安全项保留。useful-touch 中的触球后控制质量计算也保留，因此该变体只移除显式连续反馈和直接 tracking objective。

### No recovery blending

原始 raw gate、filtered gate、recovery target 和 position correction 仍按 Full 计算并写入 diagnostic。唯一变化是执行的 pelvis target 始终等于 command target，球速度 tracking reward 的 controllability factor 恒为 1。proximity reward、ball-loss、no-contact 和其他 termination 不变。

### No Stage-I initialization

环境配置与 Full 相同。差异只发生在 runner 初始化。训练脚本禁止 resume 和 migration flags，因此 actor、critic 与 LSTM 使用相同 seed 下的随机初始化。

### Interaction reference

No dense distance 仅删除 `dribbling_cg_foot_ball_distance`。No touch timing 保留 useful-touch 的轻触质量和 delayed improvement score，只让 reference window 内外的事件缩放相同。联合变体同时应用两项修改。

## 4. 训练前验证

先在 Isaac 容器内执行配置验证，不会创建仿真环境或开始训练。

```bash
/workspace/isaaclab/isaaclab.sh -p \
    scripts/rsl_rl/validate_essay13_ablation_configs.py
```

验证内容包括：

- 所有 Gym task 均已注册；
- `Ablation-Full` 除元数据外与完整 Essay13 配置一致；
- 每个单因素变体只修改目标参数；
- No recovery 保留 failure termination；
- No dense 与 No timing 没有互相误删。

## 5. 一键训练

训练脚本为：

```text
scripts/rsl_rl/run_essay13_ablation_training.sh
```

先生成命令但不启动训练：

```bash
bash scripts/rsl_rl/run_essay13_ablation_training.sh \
    --profile core \
    --stage1-load-run "<Stage-I run>" \
    --stage1-checkpoint "<Stage-I checkpoint>" \
    --stage1-migration legacy-residual \
    --seeds 13 \
    --dry-run
```

确认命令后删除 `--dry-run` 即可顺序训练 core 组。正式多种子示例：

```bash
bash scripts/rsl_rl/run_essay13_ablation_training.sh \
    --profile core \
    --stage1-load-run "<Stage-I run>" \
    --stage1-checkpoint "<Stage-I checkpoint>" \
    --stage1-migration legacy-residual \
    --seeds 13,23,37 \
    --num-envs 4096 \
    --max-iterations 100000
```

`legacy-residual` 适用于没有 bounded residual action-semantics marker 的原始 Stage-I checkpoint。若初始化 checkpoint 已经使用当前 bounded semantics，应选 `none`。`bounded-policy` 只用于旧的 unbounded reference-residual checkpoint。必须根据实际 checkpoint metadata 选择，不能混用。

只训练随机初始化变体时不需要 Stage-I 参数：

```bash
bash scripts/rsl_rl/run_essay13_ablation_training.sh \
    --variants no_stage1 \
    --seeds 13,23,37
```

`extended` profile 会在 core 之外加入 velocity observation/reward 拆分和 body/foot reference 变体。

正式训练要求 Git 工作区干净。每个 run 会在 `params/ablation_manifest.json` 保存：

- variant 和 task；
- seed、环境数、迭代数和 rollout length；
- Stage-I checkpoint 绝对路径与 SHA-256；
- 所有 motion 文件的 SHA-256；
- Git commit、branch 和 status；
- 完整启动参数。

## 6. 统一评估

所有 checkpoint 使用与 Full 相同的测试脚本和 scenario。先建立一个文本文件：

```text
variant                         load_run                               checkpoint
full                            <full run directory>                   <checkpoint file>
no_ball_velocity                <no-velocity run directory>            <checkpoint file>
no_recovery                     <no-recovery run directory>            <checkpoint file>
no_stage1                       <no-stage1 run directory>              <checkpoint file>
no_dense_distance               <no-dense run directory>               <checkpoint file>
no_touch_timing                 <no-timing run directory>              <checkpoint file>
no_interaction_reference        <no-interaction run directory>         <checkpoint file>
```

运行：

```bash
bash scripts/rsl_rl/run_essay13_ablation_evaluation.sh \
    --checkpoint-table <上面的文件> \
    --profile core \
    --eval-seeds 13,23,37 \
    --videos representative
```

评估脚本要求 soccer 源码与当前 committed HEAD 完全一致，然后为每个变体调用同一套 command grid、transition、recovery 和 long-horizon 测试。获得最终 checkpoint 后再运行 `paper` profile。

## 7. 尚未替用户决定的实验参数

以下内容不能从当前仓库可靠推断，需要在正式运行前确认：

- Essay13 实际使用的 Stage-I run 和 checkpoint；
- Stage-I checkpoint 对应的 migration mode；
- 正式训练是使用 88,000、100,000 还是其他统一迭代预算；
- checkpoint selection 使用固定末次 checkpoint 还是预先定义的 validation rule；
- 正式训练 seed 数量。

代码已经把这些参数显式化，不会静默选择 latest checkpoint。
