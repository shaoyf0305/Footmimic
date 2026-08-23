# Essay13 一键评估与消融版本管理

## 1. 一键评估脚本

入口脚本为：

```text
scripts/rsl_rl/run_essay13_baseline_suite.sh
```

它在宿主机执行时会自动调用 `$WORK/run_isaaclab.sh`，进入 Isaac 容器后逐项运行 `play_multi.py`。因此不需要手工复制多条 Isaac Lab 命令。

### 第一次运行

先执行 smoke test，检查 checkpoint、motion path、Isaac 环境和输出路径。

```bash
bash scripts/rsl_rl/run_essay13_baseline_suite.sh \
    --profile smoke \
    --videos none
```

smoke test 成功后运行 core suite。

```bash
bash scripts/rsl_rl/run_essay13_baseline_suite.sh \
    --profile core \
    --videos representative
```

脚本默认使用以下 Essay13 参数。

```text
task            Tracking-CG-G1-Dribbling-RNN-control
motion path     motions/master-v2
load run        2026-08-20_02-48-27_s2_13
checkpoint      model_88000.pt
experiment      g1_dribbling_essay
device          cuda:0
evaluation seed 13
```

如果这些参数发生变化，可以在命令行覆盖。例如：

```bash
bash scripts/rsl_rl/run_essay13_baseline_suite.sh \
    --profile core \
    --load-run "2026-08-20_02-48-27_s2_13" \
    --checkpoint model_88000.pt \
    --device cuda:0 \
    --eval-seeds 13,23,37 \
    --videos representative
```

### 三种 profile

#### smoke

只运行一个短稳态测试和一个短恢复测试。它的用途是发现 CLI、checkpoint 或输出路径问题，不能作为论文结果。

#### core

实际可用的第一轮完整评估，包括：

- 原始 36 秒 Essay13 heading sequence 回归测试。
- 串联的 3×3 速度方向网格。
- 方向切换、速度切换和速度方向耦合切换。
- 四种球位置扰动和四种球速度扰动。
- 一条 120 秒长时命令序列。

串联网格将多个命令组合放在一条长轨迹中，启动开销较小。后续分析时需要按照 `segment_idx` 切分，并丢弃每次命令切换后的初始 transient window。

#### paper

论文级扩展评估，包括：

- 25 个独立的速度方向组合。
- 三类命令切换。
- 八类受控恢复。
- 120 秒长时控制。
- 3×3 初始球偏移与 8 个参考起始相位的完整组合。

一个 evaluation seed 对应约 110 次 Isaac 启动。建议先用 `core` 确认所有测试，再让 `paper` 在空闲机器上运行。增加 `--eval-seeds 13,23,37` 会将整个 suite 对三个评估随机种子各执行一次。

### 视频选项

```text
--videos none
--videos representative
--videos all
```

`representative` 只为原始 baseline、耦合切换和一个横向恢复案例录制双视角视频。`all` 会产生大量视频，不建议在 paper profile 中使用。

### 断点续跑

如果运行中断，可以指定原来的结果目录并跳过已有 diagnostic 的案例。

```bash
bash scripts/rsl_rl/run_essay13_baseline_suite.sh \
    --profile paper \
    --result-dir output/essay13_baseline_suite/已有目录 \
    --resume \
    --videos representative
```

### 只检查命令

```bash
bash scripts/rsl_rl/run_essay13_baseline_suite.sh \
    --profile core \
    --dry-run \
    --videos none
```

`--dry-run` 会生成 suite 配置、case metadata 和所有实际命令，但不会启动 Isaac Sim。

## 2. 输出内容

默认输出到：

```text
output/essay13_baseline_suite/<UTC时间>_<profile>/
```

目录结构如下。

```text
manifest.tsv
suite_config.txt
git_status.txt
git_diff.patch
checkpoint_sha256.txt
motion_sha256.txt
source_sha256.txt
source_snapshot/
SUMMARY.txt
failed_cases.txt
cases/
  <case_id>/
    seed_<evaluation_seed>/
      case_metadata.txt
      stdout.log
      diagnostic.npz
      video/                  # 仅在启用视频时出现
```

运行结束后还会生成：

```text
<结果目录>.tar.gz
<结果目录>.tar.gz.sha256
```

之后只需要发送 `tar.gz` 和对应的 `sha256` 文件，即可统一计算论文指标和生成图表。若压缩包过大，可以先发送所有 `diagnostic.npz`、`manifest.tsv`、`suite_config.txt` 和 `stdout.log`，视频单独传输。

## 3. 新增的评估接口

一键脚本使用了 `play_multi.py` 中以下 evaluation-only 参数。

```text
--diagnostic_path
--evaluation_case_id
--evaluation_reference_phase
--evaluation_initial_ball_offset FORWARD LATERAL
--evaluation_ball_perturb_step
--evaluation_ball_position_delta FORWARD LATERAL
--evaluation_ball_velocity_delta FORWARD LATERAL
--disable_interval_pushes
--stop_on_done
--video_output_dir
```

这些参数在不传入时不会改变原来的训练和普通 playback 行为。

原始 36 秒 baseline case 保留 interval robot pushes，以复现用户提供的原始命令。其他受控测试关闭随机 interval pushes，避免它们与指定球扰动混在一起。

## 4. 消融实验不建议每个变体建立长期 Git 分支

推荐结构是一个冻结 baseline tag，加一个共同的消融实现分支，再用不同配置区分变体。

### 4.1 冻结 Essay13 baseline

Essay13 checkpoint 的训练启动时间为 2026-08-20 02:48，对应训练启动前的源码提交 `a589bd7`。从 `a589bd7` 到当前 `cfe6196`，已提交的训练与任务源码没有变化。因此 baseline 应冻结在时间戳直接对应的 `a589bd7`，而不是根据分支名称推断版本。

本地 annotated tag 和基线分支已经建立。对应命令为：

```bash
git branch baseline/essay13 a589bd7
git tag -a essay13-baseline a589bd7 \
    -m "Frozen Essay13 baseline used by model_88000.pt"
git push origin essay13-baseline
git push origin baseline/essay13
```

前两条已经完成，两个 `git push` 尚未执行。

一键脚本会比较当前 soccer 源码与 `a589bd7`。允许的差异只有默认关闭的评估起始相位接口，以及已确认保留的 actor finite-value/std-floor 安全补丁。两者都会随结果保存源码快照、SHA-256 和 Git diff。发现其他源码漂移时脚本会停止。`--allow-baseline-drift` 只用于调试，不能用于采集论文数据。

### 4.2 建立一个消融实现分支

```bash
git switch -c experiments/essay13-ablations essay13-baseline
```

所有变体共享的开关、日志和配置类都放在这个分支中。建议让以下配置同时存在，而不是反复修改同一个配置文件。

```text
essay13_full
essay13_no_ball_velocity
essay13_no_recovery_blend
essay13_no_stage1_init
essay13_no_dense_distance
essay13_no_touch_timing
essay13_no_interaction_reference
```

训练时通过 task/config 名称选择变体。这样所有变体可以在同一个 Git commit 上训练，代码差异只来自被选择的配置。

### 4.3 为什么不建议每个变体一个长期分支

如果每个消融各建一条分支，后续修复公共 bug 时需要在所有分支之间 cherry-pick。不同分支很容易逐渐出现额外差异，最终无法确认性能差异究竟来自消融因素还是代码漂移。

只有在两个变体的实现互相冲突，无法通过配置开关共存时，才考虑短期分支。短期分支完成后应合并回共同的实验分支。

### 4.4 推荐的 commit 结构

```text
commit 1  Add common ablation flags and run metadata
commit 2  Add ball-velocity and recovery config variants
commit 3  Add Stage-I initialization variant
commit 4  Add interaction-reference config variants
commit 5  Freeze evaluation and training manifests
```

所有变体经过检查后，可以给共同实现建立版本 tag。

```bash
git tag -a essay13-ablation-suite-v1 <共同实验commit> \
    -m "Config-controlled Essay13 ablation suite"
git push origin experiments/essay13-ablations
git push origin essay13-ablation-suite-v1
```

不需要为每一次训练 seed 创建新 branch 或 Git push。训练 seed 属于运行元数据，不属于源代码版本。

## 5. 每次消融训练必须记录的版本信息

建议每个训练 run 使用以下命名。

```text
e13_full_seed13
e13_no_velocity_seed13
e13_no_recovery_seed13
e13_no_stage1_seed13
e13_no_dense_seed13
e13_no_timing_seed13
```

每个 run 目录至少保存：

- `git rev-parse HEAD`。
- `git status --short`。
- 完整 resolved config。
- 相对 `essay13_full` 的配置 diff。
- Stage-I checkpoint hash。
- 初始或恢复的 Stage-II checkpoint hash。
- training seed。
- 环境数、iteration 数和训练 wall-clock time。
- checkpoint selection rule。

正式训练前工作树应保持 clean。若确实需要在 dirty worktree 上进行临时测试，必须把 `git diff` 一并保存，不能只记录 commit hash。

## 6. 多个消融并行训练时使用 worktree

如果集群任务需要在不同目录同时运行，可以让所有 worktree 指向同一个共同实验 commit。

```bash
git worktree add ../Footmimic-e13-ablations experiments/essay13-ablations
```

通常一个 worktree 配合多个配置已经足够。只有运行系统会修改工作目录内文件时，才需要为不同任务再建立多个 worktree。重点是让所有训练读取同一个 commit，而不是为每个 seed 复制一套源代码。

## 7. 推荐实际顺序

1. 确认 Essay13 checkpoint、commit 和参考数据哈希。
2. 建立 `essay13-baseline` tag。
3. 运行 evaluation smoke profile。
4. 运行 core profile 并检查 diagnostic 是否完整。
5. 运行 paper profile。
6. 建立共同的 `experiments/essay13-ablations` 分支。
7. 先实现配置驱动的 Full、No velocity 和 No recovery。
8. 用一个 seed 做短训练验证配置确实只改变目标因素。
9. 冻结 `essay13-ablation-suite-v1` tag。
10. 启动正式多 seed 训练。
