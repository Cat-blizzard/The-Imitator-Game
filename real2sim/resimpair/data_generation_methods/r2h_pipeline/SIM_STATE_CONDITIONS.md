# 已验收苹果仿真 → 锚图条件（候选 v1）

`sim_state_conditions.py` 是 CPU 条件准备入口，仅支持已验收的
`strict_apple_clutter` 苹果换碗执行。它复用 `scene_proposals` 的
投影、证据绑定和抽帧接口，不改旧生成器、物理验收或人物验收。

输入是同一次执行的 receipt、其 hash 绑定的 `verification.json`、
状态/动作数组、真实抓取与碗支撑力、同步多视角清单和所选相机。
运行时重新调用未修改的 `verify_bowl.verify`，其源文件 hash
必须与执行时捕获的一致；全部所选视频帧另查 PTS、长度和尺寸。
相机内外参从已记录 eye/target/FOV 重建核对，防止错用另一视角。

输出 `conditions.json`：

- 角色用执行 UUID、object/container index、资产 UID/mesh hash、
  世界位置绑定；颜色或画面左右不决定源和目标。
- 初态、实际抓取、整个物体清出碗沿、实际释放、末帧条件，
  每张 PNG 和源视频均保存 hash/帧索引/原时间戳。
- 每个机位分别计算源/目标碗投影位置与左右关系。
- 对象相对容器的投影 bbox 比例为参考值，不是生成图测量结果。
- 终态编辑差分为移除 `inside(object,source)`、增加
  `inside_and_stably_supported(object,destination)`，并保留身份、
  容器位置和非目标物。

关键边界：

1. RGB 为 command 前观察；`final_object_states` 为最后 command
   后的测量。终图使用实际录制末帧 `n-1`，并将这一帧状态交给原
   `verify_bowl.outcome` 单独检查，不能假借未录制的后状态。
2. `bowl_contact` 支撑力按果物聚合全部碗，不含目标碗 contact ID。
   原验收由目标几何、聚合支撑力、释放和稳定历史共同支持终态，
   不声称已有逐目标接触标签。
3. 旧记录没有导出碗的 renderer actor ID，此字段为 unknown；
   稳定 container index 身份有据可查，不能编造 segmentation ID。
4. 投影与可见像素属于仿真。生成图改变相机、布局或本体后必须
   重新确认角色和比例；这一步没有验证任何生成图或人体结构。
5. 该入口不调用 API，`generation_allowed=false`。后续仍需绑定
   获准的单人计划、检查人类首/终图，再授权收费视频生成。

开发运行：

```bash
PolicyGen/.venv/bin/python -m data_generation_methods.r2h_pipeline.sim_state_conditions \
  --receipt /mnt/sdg/datasets/real_sim_human_pipeline/harness_20261021_v1/multiview/development/episode/receipt.json \
  --verification /mnt/sdg/datasets/real_sim_human_pipeline/harness_20261021_v1/multiview/development/verification.json \
  --camera opposite \
  --output /mnt/sdg/datasets/real_sim_human_pipeline/harness_20261021_v1/state_conditions/development \
  --partition development
```

该输出已经存在，不可原地重跑。开发例的 6 个条件帧为
0、71、101、193、194、304；第194帧释放但支撑为0，未被误写成
已稳定放置。开发例 native 源在右、opposite 源在左，是从当前
投影得出，不适用于所有布局。12 项 CPU 测试通过；保留验证布局
由独立审计方执行，不以开发例推断新来源或新任务泛化。
