针对dribble作出的主要改进

已实现：
持续无球接触终止（避免不碰球）
速度跟踪奖励 （控球）
forward reward，世界坐标系 （修“绕圈”）
CG（额外的接触时序先验，接触前动作更加自然）
移除 orientation reward （适配绕桩视频）
dribble-distance，从01改成距离 （整体性优化）
物理优化-摩擦力增加 （修只踢一下）
奖励抑制 consecutive touches （修“夹球/连触”）
v1.7 防螃蟹步（回退 v1.6 后恢复）：``dribbling_face_ball`` 需球在前且骨盆朝 +X；``task_heading_alignment``；双坐标系横向惩罚；``forward_dominance`` 门控前进/追球/速度一致
已回退（v1.7–v1.13，三阶段逻辑）：
touch / chase / seek_touch / approach 分阶段速度目标
``dribbling_approach_closing``、``dribbling_phased_forward_velocity`` 等阶段奖励
阶段相关的 termination 与 play HUD

保留自 v1.9+ 的工具链（与 MDP 无关）：
``shell/video_to_npz_pipeline.sh``
``play_multi.py`` 单/双视角录制 + HUD（无阶段显示）
``dual_view_recorder.py`` 更新

下一步：
latent（AMP / VAE 动作解耦与技能库，见 proposal 2.3）
