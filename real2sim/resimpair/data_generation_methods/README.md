# 数据生成方法与参考视频研究

本目录是方法、提示词、研究依据和试验设计的统一入口。创建于2026-10-19。

- [2026-10-21 管线机制与楷体新版 PPT](PIPELINE_EXPLANATION_AND_REDESIGN_20261021.md)：20 页解释数据／窗口／合同／资产能力／按需相机／工具验收与修复；重新清点全部 14 条生成视频、10 张锚图。55,535 条现存模型路径不等于交互资格；固定对面＋风险标签是建议，未改默认门。
- [2026-10-21 五例 PPT、时间费用与双臂／灵巧手实测](REPORT_AND_CONTROL_RESULTS_20261021.md)：14 页、7 段实际视频；两个 planner 同步执行和 Panda＋Allegro 的 The-Imitator-Game 后端已接通，稳定抓持均未通过。只读查询获得当前报价，历史实际扣费仍未知；旧生产默认不变。
- [2026-10-21 续研：单人几何、左右手/视角对照与隐藏内容条件](HARNESS_RESULTS_20261021_V3.md)：同一模型区分固定右手失败与左手必要条件成立，完成实际准备入口串源修复和两柜门布局条件验证；旧失败与几何投影差异保留，新API/GPU/有效合成人类配对均0。
- [2026-10-21 续研：统一首图入口、累计预算修复与隐藏物体反例](HARNESS_RESULTS_20261021_V2.md)：真实来源/状态/人物接线及独立反例验证，724项测试及13项子测试；新API/GPU均0，隐藏物体比例诊断保留漏检，不宣称视觉生成改善。
- [2026-10-21 接手实测：同步多视角、单人前置约束与失败回归](HARNESS_RESULTS_20261021.md)。长期接手入口为根目录 [START_HERE.md](../START_HERE.md)、[HARNESS.md](../HARNESS.md) 与 [CURRENT_STATE.json](../CURRENT_STATE.json)；候选未晋级默认，未付费生成、未训练。
- [最新：路径修复、家居扩增与预算受限对照（2026-10-20E）](PIPELINE_REPAIR_AND_CONTROLLED_20261020E.md)：15次最终严格执行、105段路径独立复查、家居布局冲突修复、同轨迹视角实验、三任务视频对照及实际失败；未训练。
- [最新方法与自动化实测（2026-10-20D）](PIPELINE_METHOD_20261020D.md)：统一为TaskTrace／SceneForge／EventGuard；真人盲任务提取、几何候选＋VLM修正、质量评分反例、4个真实随机化试点与明确边界。
- [Agent运动规划规范](human_to_robot_pipeline/PLANNING_GUIDE.md)：标准接口、任务保持、物理杂物、关键事件可见性和独立验收。
- [真人任务草案](HUMAN_TASK_DRAFT_20261020D.md)、[自动关键帧与ROI](AUTO_SCENE_PROPOSALS_20261020D.md)、[质量模型与人物方案约束](QUALITY_CASCADE_20261020D.md)：代码、真实输入输出与剩余人工项。
- [最新关键状态条件与事件纠错实测（2026-10-20C）](STATE_CONDITIONED_RESULTS_20261020C.md)：真实首尾图条件、有限自动调度、新旧193帧局部工具检查和24帧独立视频复核；明确未接入的自动分支与训练边界。
- [详细PDF报告](/mnt/sdg/datasets/real_sim_human_pipeline/state_conditioned_20261020c/report/Real2Sim2Human_pipeline_20261020.pdf)与[完整报告文件夹](/mnt/sdg/datasets/real_sim_human_pipeline/state_conditioned_20261020c/report/)：真实输入、旧问题、中间关键图、改进视频、模块触发和证据；视频为可复制的普通文件，不是软链接。
- [批量实测（2026-10-20第二批）](PIPELINE_BATCH_RESULTS_20261020B.md)：4条候选、补帧重审、柜门隐藏状态缺口、4条显式配对与12次实际CPU读取。
- [当前完整架构](PIPELINE_ARCHITECTURE_20261020B.md)：已实现、实际跑通与尚未接通的边界。
- [训练数据接口审计](TRAINING_DATA_AUDIT_20261020B.md)：原读取器的问题、隔离适配器与真实数值检查。
- [最新实测与论文方法计划（2026-10-20）](REAL_SIM_HUMAN_EXECUTION_PLAN_20261020.md)：Echoflow接口实测、真实任务→仿真→合成人类参考→真实执行的分工、关键缺口与受控实验。
- [管线问题、针对性修复与实测（2026-10-20）](PIPELINE_REPAIR_RESULTS_20261020.md)：当前主线，不启动训练；原始失败、显式帧审查对照及生成修复状态。
- [方法贡献和实验设计审查](PIPELINE_FAILURE_METHOD_REVIEW_20261020.md)：区分已发现问题、潜在风险与尚待验证的论文主张。
- [中转配置示例（无key）](config.echoflow.example.json)：Gemini/Nano原生接口与Seedance异步接口。

- [运行第一版流水线](RUN_PIPELINE.md)：可执行入口、实际6条多目标输入、API配置、恢复机制和当前未完成项。
- [真实人类窗口→仿真续采](human_to_robot_pipeline/README.md)：2026-10-20恢复严格人类任务对应的有限批量生成、源绑定和独立验收；与合成人类视频路线分开。
- [API需求](API_REQUIREMENTS.md)：两个平台的三个模型权限、准确ID和素材限制。
- [持续反思与当前判断](../DATA_GENERATION_REFLECTIONS.md)：后续直接修订，并保留简短变更记录。
- [数据链路与使用约定](PIPELINE.md)：任务定义、批量筛选、规划、质量、数据路由与发布。
- [数据生成goal提示词](prompts/DATA_GENERATION_GOAL.md)：供后续明确启动任务时使用。
- [机器人转人类参考研究](ROBOT_TO_HUMAN_REVIEW.md)：论文、开放实现、接口能力、用途边界及改进方向。
- [完整链路设计与批判性审查](REFERENCE_PIPELINE_DESIGN.md)：复用现有同初态多目标数据，任务信息验收、长程转换、纠错、Agent/工具分工与W行为验证；截至2026-10-19的设计稿。
- [小规模试验计划](ROBOT_TO_HUMAN_PILOT.md)：输入、对照、质量评价及后续受控训练方案。
- [转换与审核提示词](prompts/ROBOT_TO_HUMAN_PROMPTS.md)：任务事实提取、初始图、操作视频和独立审核。
- [Zero-WAM公开清单实测证据](sources/zero_wam_20261019/SOURCE_INDEX.json)：2499配对/994人类路径，附可复算脚本；没有下载大视频包。
- [原生试验输入](pilot_inputs/README.md)：3类任务各2条，18张原生预览，12次完整视频解码与源文件哈希已验证；尚未生成合成人类视频。

## 现有代码与数据入口

| 功能 | 当前路径 |
|---|---|
| 人类/机器人任务库、配对与原生窗口读取 | [hypericl_task_library](../PolicyGen/examples/baselines/hypericl_task_library/README.md) |
| 最近的人类窗口优先采集 | [generation](../PolicyGen/examples/baselines/hypericl_task_library/generation/README.md) |
| 碗、书、遥控、柜门等数据工厂 | [hypericl_data_factory](../PolicyGen/examples/baselines/hypericl_data_factory/README.md) |
| 骨牌、积木、接口等扩充 | [hypericl_ego_task_expansion](../PolicyGen/examples/baselines/hypericl_ego_task_expansion/README.md) |
| 任务与配对库 | `/mnt/sdf/datasets/hypericl_human_robot_task_library_20261019_v1` |
| 全部任务报告 | `/mnt/sdf/datasets/hypericl_sim_generation_report_20261019_v1` |
| 完整人机配对交付 | `/mnt/sdf/datasets/hypericl_human_sim_pairs_report_20261019_v1` |

已有运行代码和冻结数据没有移动，避免破坏路径与历史复现。本目录先统一方法与新研究，后续重构采用明确版本，不靠复制旧采集器制造多个入口。

## 状态

当前状态见2026-10-21报告和根目录 CURRENT_STATE。新规则复核确认旧苹果/书本首图及5条派生均双人；旧局部“支持”不包含单人验收。同步多视角已在两次苹果物理执行中实测，原视角保留；新 API 调用0，候选未晋级。

以下为2026-10-20E历史：独立strict入口已验证，最终15次原生执行通过，14个视觉配置、6组不同动态轨迹；105段路径、4924个加密点通过独立重建碰撞复查。12类家居资产完成预检、8类实际进入仿真；8个家居配置修复后通过。同轨迹换视角使闭爪可见帧由0/16变为16/16，当时为替换相机并重新执行。三来源视频对照首轮A/B各2/3获当时局部可见支持；后验修终图后B为3/3，A重抽仍2/3，不能解释为统计优势或单人合格率。总图像4／视频8／审核8次；605项软件测试通过。未训练、未正式发布，媒体仍在SDG。

以下为2026-10-20D历史：完成1个真人窗口的盲任务草案、4个场景的自动候选提取、新柜门193帧自动ROI下游诊断；8条视频的CPU质量检查、2次匿名VLM评分和人物方案约束检查。4个扩大布局的物理候选中2条带额外静态杂物的执行通过原生验收，2条失败保留。

D阶段随后查明自定义Screw规划器未使用碰撞结果拒绝路径，且缺少终点残差校验。当时新增离线FK路径检查、保留4条原始执行与精确源码归档，并停止旧实验入口。E阶段已实现隔离的strict执行器，不再处于仅离线检查状态；旧生产代码和旧失败归档均未修改。

以下为上一批历史：3条新视频＋1条复用，覆盖3个独立真人任务窗口；4条跨执行显式配对的12次CPU读取通过。前三例任务过程可读，柜门存在计划外手和隐藏对象保留不确定。正式质量发布及训练均为0。旧生产loader尚未切换到隔离适配器。

当前优先研究管线，不启动训练。后续修复已完成同视频的显式均匀8帧/边界8帧审查对照，两者均检出疑点，后者定位更精确；新增密集候选探针和固定预算抽帧，代码及未覆盖证据保留。另一次固定角色prompt生成已经返回，新样保持初态，完成从右碗到左碗的搬运并收手；作为任务语义候选通过，尚非正式质量发布，也不是统计成功率。状态见 `/mnt/sdg/datasets/real_sim_human_pipeline/repair_study_20261020_v1/WATCH_STATUS.json`。详细结果和局限见上方修复报告。

以下为本日首轮接入情况：

已完成第一版流水线软件与CPU输入准备：另接入2个完整三目标组共6条长程轨迹，96张事件帧，提供可恢复提交/查询、盲审及配对候选导出。2026-10-20接入用户中转站：Gemini真实人类视频观察和Nano人类锚帧编辑实测成功；Seedance一次提交超时后只读恢复取得8.042秒视频，未重发。视频有初始苹果瞬移和错误终态，已经拒收；VLM审核也存在格式不合格与漏掉瞬移问题，保留原响应供诊断。新增通用任务证据门控和苹果真实源契约，完整软件测试165项通过。新生成数据仍未进入训练。SAM/CoTracker接入、深度mask补录、通用契约接生产reader和长视频连续扩展仍为待实施项。产物见 `/mnt/sdg/datasets/real_sim_human_pipeline/pilot_20261020/`。
