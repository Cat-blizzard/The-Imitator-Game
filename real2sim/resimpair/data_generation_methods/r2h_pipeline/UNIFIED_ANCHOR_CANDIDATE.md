# 来源、人物和初态条件统一入口

`unified_anchor_candidate.py` 是新的 opt-in 候选，未替换旧生成器。它连接已存在的单人计划、仿真状态条件、真实人来源与实际图像/视频 provider；不修改独立人物或静态状态验收规则。

## 此次补上的实际缺口

- 单人 v2 计划里的 `human_source_id` 和 `physical_execution_id` 原先只检查非空，不能证明是这次执行的来源。
- 同一个任务契约可以列出多个真人窗口。本执行来源来自 receipt 中的 `structured_proposal.human_source_window`，不能取契约列表第一项，也不能用任务名代替来源会话。
- 仿真条件有相机专属角色方向和终态差分，但旧单人请求没有消费它。把整个差分直接放入首图请求会要求图像提前完成任务。
- 旧单人视频入口只负责人物门；人物通过不能推断对象、比例、角色或初态通过。
- 旧预算按输入预算文件路径创建账本；复制配置可重置计数。新候选改用固定 campaign 注册表及稳定账本，此修复不追改旧执行记录。

## 请求如何绑定

`prepare` 接收 `conditions.json`、本次执行 `input.json`、人物 placement 和 provider 配置。它会：

1. 核对条件包、receipt 与 execution input 的 UUID、seed、split 和完整 proposal；重新从原 receipt、物理验收及所选相机编译条件，与原包逐字段比较。临时导出图的路径可不同，图像内容、视频、帧号、时间、状态、角色、相机及条件均须一致。
2. 从 proposal 精确选择真人窗口，核对 task spec/contract 中对应成员、原视频和审核图 hash、会话、窗口、FPS、split、实际解码长度和 semantic review 对应条目。
3. 根据稳定执行角色和选相机建立任务契约，再将这个来源及初图绑定到一人身体/双手/事件计划。操作者位置仍为显式设计，未声称自动姿态或人体可达性估计。
4. 输出自带内容 hash 的 `package.json` 和 `initial_request.json`。首图载荷只含人物位置与初始静止手角色、初态关系、同相机角色和尺寸参考；不包含未来事件分配、终态图或 `state_delta`。

`validate` 和 `execute-initial` 都重新执行上述来源核验。执行时请求必须与当前包重新构造的请求完全相同。输出图像绑定实际 campaign stage、provider 结果、请求和 package；缓存恢复不会重新 POST。

## 两个独立门与可运行阶段

图像生成使用现有 `GeminiClient.generate_image`。视频准备/提交使用现有 `SeedanceClient`，但必须同时有：

- 与当前图像、计划绑定的独立 `single_actor_v2.1` 人物证据。
- 与当前图像、package、来源、相机、角色和初态绑定的独立 `static_state_v1.0` 证据。
- 本入口生成回执，且实际图像像素与已保存 provider 结果一致。

缺失、失败和 unknown 都阻断。静态门单独返回的 `can_generate_video` 恒为 false；统一入口组合人物门和 `clears_static_state`。这些条件只允许尝试视频，不能批准全片任务、动作接触、连续性、配对发布或训练。

```text
python -m data_generation_methods.r2h_pipeline.unified_anchor_candidate prepare
  --conditions CONDITIONS --input EXECUTION_INPUT --placement PLACEMENT --output NEW_DIR --config CONFIG
python -m data_generation_methods.r2h_pipeline.unified_anchor_candidate execute-initial
  --package PACKAGE --config CONFIG --budget BUDGET --credentials PRIVATE_ENV
python -m data_generation_methods.r2h_pipeline.unified_anchor_candidate video-gate
  --package PACKAGE --image IMAGE --actor-review ACTOR_REVIEW --state-review STATE_REVIEW
python -m data_generation_methods.r2h_pipeline.unified_anchor_candidate prepare-video
  --package PACKAGE --image IMAGE --actor-review ACTOR_REVIEW --state-review STATE_REVIEW --config CONFIG
python -m data_generation_methods.r2h_pipeline.unified_anchor_candidate submit-video
  --package PACKAGE --config CONFIG --budget BUDGET --credentials PRIVATE_ENV
python -m data_generation_methods.r2h_pipeline.unified_anchor_candidate poll-video
  --package PACKAGE --config CONFIG --credentials PRIVATE_ENV
```

命令示意中的换行用于排版；实际运行时参数应与命令在同一 shell 命令中。

## 预算与验证边界

新预算 schema 为 `r2h_unified_budget_v1`：必须显式提供 `campaign_id`、绝对 `ledger_path`、累计 image/video/review 调用上限、并发 1 和最多 600 秒单请求超时。固定注册表位于本轮 `harness_20261021_v2/campaign_registry`，同 ID 不允许改账本或提额。预算副本使用账本内部同一规范配置；复制路径既不产生新配额，也不会把已有请求误认成新请求。未知提交继续占额、阻断新调用；异步视频直至查询终态才释放槽位。

实际 g00 统一包位于：

`/mnt/sdg/datasets/real_sim_human_pipeline/harness_20261021_v2/unified_anchor/development/package.json`

当前预算全部为 0。`zero_budget_and_missing_reviews.json` 记录实际 provider 入口在网络前被预算阻断；视频因缺生成回执和两份独立审核而阻断，付费 call marker 为 0。

20 项本地测试覆盖初态隔离、包/相机/系谱篡改、provider 参数接线、两门独立阻断、像素来源、预算复制、unknown 占额及 campaign 改额/改账本。测试 provider 响应和审核是明确标记的人工 fixture，不是新生成质量结果。真实条件/系谱与 g01 由独立审计另行验证。

当前未产生新合成人类图像、视频或有效配对。下一步需明确新预算后先做一张开发首图，通过两个独立静态门再做视频；其他真人来源的效果仍须实际生成验证。
