# Essay13 Baseline 固定与代码审计

## 1. 已确认的基线身份

```text
run             2026-08-20_02-48-27_s2_13
checkpoint      model_88000.pt
source commit   a589bd71168bd876fe4db93a3d887039c94005a8
local branch    baseline/essay13
annotated tag   essay13-baseline
```

训练 run 在 2026-08-20 02:48 启动，`a589bd7` 的提交时间为同日 02:45。它是启动前直接对应的源码提交。`a589bd7..cfe6196` 在 `source/whole_body_tracking/soccer` 下没有已提交差异，因此不需要用较晚的提交代替时间戳对应版本。

本地 branch 和 tag 均指向 `a589bd7`。尚未执行 remote push。

## 2. Baseline 中实际存在的主要组件

以下结论来自 `a589bd7` 的 Stage-II task 配置和实现代码。

- 任务是连续速度、方向和命令持续时间控制。
- 训练命令范围为速度 0.40--1.65 m/s、方向 -0.75--0.75 rad、持续时间 3--6 s。
- 完整 reward 中包含球速度追踪、过快惩罚、闭环球位置控制和 recovery gate。
- 完整 reward 中包含身体位置、双脚位置和参考触球距离等动作与交互参考项。
- 动作接口保持 29 维，其中 14 个上肢维度使用有界 reference-relative residual。
- Essay13 的上肢实现确实包含 rank-6 PCA manifold、0.10 rad orthogonal residual bound、1.8 Hz low-pass filter 和 0.25 rad reference envelope。
- 上肢 policy distribution 对相应动作维度使用 tanh-transformed Gaussian，并在 PPO likelihood 中计算变换修正。

最后两项说明 Essay13 并不是只有一个简单 tanh 的 direct-v3 上肢实现。将 Essay13 作为正式 baseline 时，应把它与后续 direct-v3 版本区分开，不能把 direct-v3 的 Method 描述直接用于解释该 checkpoint。

## 3. Ablation 检查

当前冻结源码没有注册或配置任何正式消融变体。

- `source` 中只注册 Stage-I mimic task 和 Stage-II full dribbling task。
- 没有 `no_ball_velocity`、`no_recovery`、`no_stage1_init`、`no_motion_reference` 或 `no_interaction_reference` 等 task/config。
- 一键 baseline suite 只改变测试命令、初始条件和外部球扰动。它不改变 reward、observation、action、policy 维度或训练配置。
- `play_multi.py` 在原始 baseline 中已有若干默认关闭的 play-only counterfactual 参数。suite 不会传入这些参数，因此它们不会改变 baseline rollout。它们也不是训练消融版本。

后续消融应从 `essay13-baseline` 建立独立实验分支，并通过并列配置选择变体。不要在 `baseline/essay13` 上直接提交消融实现。

## 4. 当前工作区与冻结基线的差异

当前 checkout 仍是 `essay`，没有切换到 `baseline/essay13`。这样可以避免覆盖现有未提交工作。

允许的 evaluation-only 改动为：

```text
scripts/rsl_rl/play_multi.py
source/whole_body_tracking/soccer/tasks/tracking/mdp/commands_multi_motion_soccer.py
scripts/rsl_rl/run_essay13_baseline_suite.sh
```

其中 command 配置新增的固定起始相位字段默认为 `None`。没有传入 evaluation 参数时，它不改变原始行为。

另保留以下运行时安全补丁：

```text
source/whole_body_tracking/soccer/utils/bounded_actor_critic.py
```

该改动来自另一版本的运行经验，加入 exploration standard deviation floor 和 NaN/Inf 检查。按照当前实验约定，它作为不影响确定性策略输出的安全补丁保留，不视为 ablation。一键脚本允许这一项差异，并将文件快照、SHA-256 和完整 Git diff 保存到每个结果包中。其他未批准的 soccer 源码差异仍会使脚本停止。

## 5. 仍需由运行产物确认的内容

源码可以确认 task 的默认配置，但不能单独证明当时训练命令实际使用了哪些覆盖参数。开始消融训练前，最好从 Essay13 run 目录补充保存以下文件：

- resolved environment config；
- resolved agent config；
- 原始训练命令或 stdout；
- Stage-I initialization checkpoint 的路径和 SHA-256；
- `model_88000.pt` 的 SHA-256 和 checkpoint metadata；
- motion bank 文件清单与 SHA-256；
- 训练 seed、环境数量和实际迭代数。

在这些运行元数据可用之前，可以确认源码基线和 checkpoint 身份，但不能仅凭仓库无条件断言 Stage-I 初始化来源等运行时事实。

## 6. 推荐的后续分支关系

```text
a589bd7  essay13-baseline / baseline/essay13
   |
   +-- experiments/essay13-evaluation
   |     只包含默认关闭的评估与诊断接口
   |
   +-- experiments/essay13-ablations
         后续加入共享消融开关和并列配置
```

baseline tag 保持不动。所有消融从同一个 tag 开始，最终在同一实现提交上通过配置区分变体。
