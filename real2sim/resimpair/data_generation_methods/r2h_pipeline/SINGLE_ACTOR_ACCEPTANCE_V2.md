# 单人结构独立验收（2026-10-21）

当前规则为 `single_actor_v2.1`。这是一层可执行的证据验收门，不是自动人体检测器。生成器提交的结构计划、图像中可见的身体和实际验收结果分别保存。旧 `anchor_body_review.py` 的两手预算只完成手数量盘点，不能推导一人。

`single_actor_acceptance.py` 独立于候选生成模块，执行六项检查：人数、解剖学左手归属、右手归属、可见身体结构、人物位置、任务操作角色。每项为 `pass/fail/unknown`。必须有绑定实际媒体的直接看图／人工观察支持，VLM 单独的肯定或检测器手数会降为 `unknown`；任一明确失败即拒绝，任一未知即停止补证据。人体不完整可见时不捏造隐藏肢体身份，也不要求合成人类 RGB 提供 3D 真值。

输入由 `schemas/single_actor_review_v2.schema.json` 实际校验，禁止未知字段和缺项。随后校验媒体路径、SHA256、人物计划 SHA256、证据引用和媒体实际可解码性。视频证据帧号必须小于 `ffprobe -count_frames` 获得的实际帧数，静图只能引用帧 0。额外开销为本地哈希和解码，不调用模型或 GPU。

```python
from data_generation_methods.r2h_pipeline.single_actor_acceptance import gate_anchor
decision = gate_anchor(review_path, anchor_path, plan_path)
if not decision["can_generate_video"]:
    # reject: 修首图；needs_review: 补足观察。两者均不提交视频请求。
    return decision
```

`can_generate_video` 只说明首图通过人物结构门，不授予总体质量发布、训练准入或物理正确标签。历史重审可以明确填写 `plan_sha256: null` 来拒绝已观察到的问题；没有计划绑定时绝不据此授予新视频生成许可。视频采样审查即使通过，也不宣称全片人物连续性通过。

本轮回归先实际查看苹果、书本共同首图，以及八条存档视频各七帧。两张首图及其五条派生视频均可见两侧独立躯干／衣袖和手臂来源，单人要求失败；苹果后验修终图的视频也仍有两人。柜门三条只显示一只手／前臂，人物完整归属仍为未知。柜门 A 原片和重抽都把小立方体变成明显较大的彩色格板，B 在检查帧保持小物体的相对尺度，但不能据此确认全片或单人结构。

实际证据和可重复入口：

- [当前冻结回归清单](/mnt/sdg/datasets/real_sim_human_pipeline/harness_20261021_v1/regression/revision_v2_1/manifest.json)：精确绑定旧视频、两张首图、原审查、56 张原尺寸采样帧、规则和源码；全部是已知开发样本。
- [回归运行结果](/mnt/sdg/datasets/real_sim_human_pipeline/harness_20261021_v1/regression/revision_v2_1/run_01/result.json)：十份人物观察重放及两条视频的真实 CPU 边界检测。人物标注重放不是自动检出率。
- [最初 v2.0 冻结证据](/mnt/sdg/datasets/real_sim_human_pipeline/harness_20261021_v1/regression/manifest.json)：保留。v2.1 修复了手工字段校验与 schema 不完全一致、未检查实际视频帧号上界的问题，视觉标准和所有样本标签未变。

```bash
PolicyGen/.venv/bin/python -m unittest \
  data_generation_methods.tests.test_single_actor_acceptance \
  data_generation_methods.tests.test_failure_regression -v

# 输出必须是新目录。
PolicyGen/.venv/bin/python -m data_generation_methods.r2h_pipeline.failure_regression \
  --manifest /mnt/sdg/datasets/real_sim_human_pipeline/harness_20261021_v1/regression/revision_v2_1/manifest.json \
  --output /mnt/sdg/datasets/real_sim_human_pipeline/harness_20261021_v1/regression/revision_v2_1/run_02 \
  --run-boundary
```

版本比较必须用同一规则重新验收基线和候选。苹果原片与历史修复片在 v2.1 下均因两人而拒绝；修复片的末帧回跳消失这一历史局部结论仍成立，两者不能互相抵消。书本和柜门早已参与已知问题诊断，不能现在改称保留验证集。

清单只按元数据保留了杯盖来源 `ego_part1_add_remove_lid_976` 作为下一轮验证候选，本轮未查看该素材。它曾有历史任务语义审核，不能声称全项目从未见过；还没有对应本轮新仿真布局和合成人类候选。实际保留验证数为 0。新来源整个会话和同一物理执行的所有派生必须留在同一研究划分。
