"""ManiSkill 3.0.1's motion-planning solver stack (base / two-finger gripper / Panda) and its OBB grasp utilities.

Copied from ManiSkill 3.0.1 (mani_skill/examples/motionplanning/{base_motionplanner, two_finger_gripper, panda}),
Apache-2.0, https://github.com/haosulab/ManiSkill; only the import paths were changed. The Imitator fork of ManiSkill
(3.0.0b21) ships an older, standalone PandaArmMotionPlanningSolver with a different API (e.g. no gripper_state
argument); the 1.0 Imitator motion-planning solutions keep using the fork's solver, the imitator single-arm Panda
experts (planning/panda_ms.py, planning/topo_tasks.py) use this copy so that their behaviour matches the platform they
were validated on.
"""
