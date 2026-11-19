# 单人首图候选 v2

这是可执行、尚未晋级的候选入口，复用现有任务契约、Gemini/Seedance 客户端、请求回执和下载器。旧默认入口与归档媒体保持原样。

它修复的是生成前条件和验收接线缺口：旧提示把机器人两侧的手臂入口保留下来，旧审核只数手。两个人各伸一只手仍然满足“两手”，因此不能用旧审核批准单人视频。

## 输入、输出和停止条件

1. `actor_plan.json` 在首图生成前固定唯一操作者、身体位置与朝向、左右手归属、各手初始位置/角色、每个必要事件的操作手。绑定实际仿真首图、真人任务契约及来源划分。允许人手入口与机器人不同；两个机器人不代表两个人。
2. `prepare-anchor` 校验计划并构造请求。缺计划、手归属不明、两人分担双手、漏掉必要事件、来源划分改变都会停止。此时仅验证声明自洽，不证明人体姿态可实现。
3. `generate-anchor` 只执行一个图像阶段。图像产生后状态为 `needs_independent_v2_anchor_review`，不会自动进入视频。
4. `prepare-video` 调用独立的 `single_actor_acceptance.gate_anchor`；图像 hash、计划 hash 和六项观察证据都必须匹配。旧手数报告、VLM 单独肯定、无关媒体证据、未知归属不能放行。这里只通过静图人物门，仍不代表对象或全片任务质量通过。
5. `submit-video` 再次核对完整请求及独立验收，防止准备之后换图、换计划、换证据。`poll-video` 仅查询已有任务并下载，不重新 POST。

独立验收代码由单独审计工作流维护；生成器不修改其规则。当前版本为 `single_actor_v2.1`，增加同源 schema 校验及真实媒体帧数核验，先前 v2.0 资格记录保留。

## 已有本地资格证据

`tests/test_single_actor_candidate.py` 有 19 项 CPU 接线测试，包括两人两手、未知归属、旧手数报告、VLM-only、图片/计划/证据 hash 错配、视频提交前图像改变、零预算、跨输出目录累计、失败占额和异步并发槽。加上独立规则及原 batch/controlled 回归，当前 v2.1 下 56 项通过。

实际苹果、书本、柜门的既有源图与契约已生成三个离线计划和请求：

`/mnt/sdg/datasets/real_sim_human_pipeline/single_actor_20261021_candidate_v2/preparation.json`

同目录 `runtime_qualification_final.json` 绑定归档的精确源码，并记录真实零预算执行阻断、苹果/书本历史首图按 v2.1 拒绝后无法进入视频请求。早期资格文件保留，当前复查使用 final 记录。

这些都是看过历史问题的**开发来源**，不是保留验证集。身体位置为检查源图后的明确设计选择，不是自动姿态/可达性估计。没有生成新图片或视频，没有视觉效果改善结论。

## 可重复执行

```bash
PolicyGen/.venv/bin/python -m data_generation_methods.scripts.prepare_single_actor_20261021 \
  --manifest /mnt/sdg/datasets/real_sim_human_pipeline/batch_20261020b/input_manifest.json \
  --config data_generation_methods/config.echoflow.example.json \
  --output /mnt/sdg/datasets/real_sim_human_pipeline/single_actor_20261021_candidate_v2

PolicyGen/.venv/bin/python -m pytest \
  data_generation_methods/tests/test_single_actor_candidate.py \
  data_generation_methods/tests/test_single_actor_acceptance.py \
  data_generation_methods/tests/test_batch_pipeline.py \
  data_generation_methods/tests/test_controlled_experiment.py -q
```

每阶段入口是：

```text
python -m data_generation_methods.r2h_pipeline.single_actor_candidate
  prepare-anchor --output CASE_DIR --config CONFIG --plan ACTOR_PLAN
  generate-anchor --output CASE_DIR --config CONFIG --budget CAMPAIGN_BUDGET --credentials PRIVATE_ENV
  prepare-video --output CASE_DIR --config CONFIG --anchor IMAGE --review INDEPENDENT_V2_REVIEW
  submit-video --output CASE_DIR --config CONFIG --budget CAMPAIGN_BUDGET --credentials PRIVATE_ENV
  poll-video --output CASE_DIR --config CONFIG --credentials PRIVATE_ENV
```

当前共享 `budget.json` 的 image/video/review 上限全为 **0**。未配置新授权预算时付费入口在读取凭据和调用网络前停止。所有候选必须使用同一个 campaign 预算；输出目录改变不重置账本。当前实现固定并发 1，每次请求超时最多 600 秒。异步视频 POST 返回后仍占并发槽，直至查询到终态；未知提交继续占额并阻断新调用。实际费用仍需账单核对。

## 下一步与晋级条件

获得明确预算后，先生成一张苹果开发首图，按固定 v2 规则独立看图。失败时先修人物方案/首图，未知时补可见归属证据；不花视频预算传播错误首图。若合格，再对不同真人来源和布局执行小批保留验证；已知书本/柜门故障可作回归，但不能改称盲测。

完整实验仍需保持任务对象、隐藏内容和关键事件的独立检查。计划字段与门控只能阻止已知或未知的不合格首图进入本候选视频阶段，不能保证生成模型遵守计划，也不能保证首图通过后整片仍只有一个人。尚无关键点/深度/姿态条件输入、人体可达性验证或自动单人视觉检测器。没有这些新证据前，不替换默认管线。
