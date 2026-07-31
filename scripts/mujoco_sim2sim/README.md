# Footmimic v4.4 control: MuJoCo sim2sim

This runner targets the preserved task:

`Tracking-CG-G1-Dribbling-RNN-control`

It reproduces the task's 172-dimensional observation, two-layer LSTM state,
29-dimensional action order, upper-body motion-bank PCA projection/filter, and
50 Hz policy / 200 Hz PD-control timing. The implementation was adapted from
the experiment code under `../HumanoidSoccer/exp/mujoco_soccer`, with the v4.4
control-specific paths replacing its generic observation padding.

## 1. Export the checkpoint

Run this in the environment containing PyTorch and ONNX:

```bash
source /home/ubt/miniconda3/etc/profile.d/conda.sh
conda activate isaaclab_211
cd /mnt/syfssd/data/Footmimic
python scripts/mujoco_sim2sim/export_v44_control.py
```

The default output is:

```text
logs/rsl_rl/g1_dribbling/2026-07-29_03-58-23_v4.4/exported/policy_93000_v44_control.onnx
```

This export does not start Isaac Sim. It reconstructs the actor/LSTM directly
from `model_93000.pt` and embeds the checkpoint observation normalizer.

## 2. Run MuJoCo

First run a short headless smoke test:

```bash
conda activate footmimic-mj
cd /mnt/syfssd/data/Footmimic
python scripts/mujoco_sim2sim/run_v44_control.py \
  --policy logs/rsl_rl/g1_dribbling/2026-07-29_03-58-23_v4.4/exported/policy_93000_v44_control.onnx \
  --motion-bank motions/master-modified \
  --motion master_001675_001884_unitree_g1 \
  --speed 0.55 \
  --heading 0.0 \
  --sim-time 2
```

Then add `--render` for the interactive viewer:

```bash
python scripts/mujoco_sim2sim/run_v44_control.py \
  --policy logs/rsl_rl/g1_dribbling/2026-07-29_03-58-23_v4.4/exported/policy_93000_v44_control.onnx \
  --motion-bank motions/master-modified \
  --motion master_001675_001884_unitree_g1 \
  --speed 0.55 \
  --heading 0.0 \
  --sim-time 0 \
  --render
```

The rendered simulation starts paused and stays open until the viewer window is
closed:

- `Space`: run or pause.
- `R`: reset the robot, ball, style phase, action filter and LSTM, then pause.
- `--sim-time 0`: disable the episode time limit.
- `--no-start-paused`: start running immediately.
- `--no-track-camera`: disable the default pelvis-following camera.

With a positive `--sim-time`, reaching the limit only pauses the simulation; it
does not close the viewer. Press `R` to start a fresh rollout.

## Optional Isaac-aligned dynamics

The original Unitree MJCF and MuJoCo's default material mixing do not reproduce
several properties of the Isaac task. Add this flag to enable the deterministic
Isaac-aligned mapping:

```bash
python scripts/mujoco_sim2sim/run_v44_control.py \
  --policy logs/rsl_rl/g1_dribbling/2026-07-29_03-58-23_v4.4/exported/policy_93000_v44_control.onnx \
  --motion-bank motions/master-modified \
  --motion master_001675_001884_unitree_g1 \
  --speed 0.55 \
  --heading 0.0 \
  --sim-time 0 \
  --isaac-aligned \
  --render
```

The aligned profile applies:

- Robot joint `frictionloss=0`, matching the source URDF.
- Foot-ground sliding friction `0.7125`, the expected effective coefficient of
  the Isaac startup randomization: `0.95 * mean([0.3, 1.2])`.
- Ball-ground sliding friction `0.589`: `0.95 * 0.62`.
- Ball-robot sliding friction `0.465`: `mean([0.3, 1.2]) * 0.62`.
- Ball mass `0.4 kg`, radius `0.11 m`, and solid-sphere inertia
  `0.001936 kg m^2`.
- PhysX-style linear and angular velocity damping rates of `0.18 1/s`.

PhysX damping is a velocity decay rate, whereas MuJoCo free-joint damping is a
generalized force coefficient. The runner therefore sets the ball joint damping
to zero in aligned mode and applies mass/inertia-scaled damping forces before
every physics step. Directly setting MuJoCo joint damping to `0.18` over-damps
the ball rotation by roughly three orders of magnitude.

This mode is opt-in. Without `--isaac-aligned`, the runner retains the previous
legacy MJCF behavior exactly. PhysX has separate static/dynamic friction and a
direct restitution coefficient while MuJoCo does not; the aligned profile maps
the effective dynamic coefficients and uses the existing low-bounce contact
constraint. Exact per-run foot friction would additionally require exporting
the random material bucket selected by Isaac.

The full `master-modified` directory must remain available even when one motion
is selected: v4.4 fitted its six-dimensional upper-body PCA manifold using all
valid frames in the motion bank.

Useful controls:

- `--ball-forward 0.45 --ball-lateral 0.0`: initial ball offset in task/world axes.
- `--destination 0 -5`: destination observation; `(0,-5)` is the training center.
- `--start-frame N`: initial and style-loop phase.
- `--heading RAD`: absolute task/world heading, not pelvis-relative heading.
- `--no-stop-on-fall`: continue a rollout after the pelvis falls below the threshold.

The generated MJCF is temporary. The source `g1_actuator.xml` is not modified.
