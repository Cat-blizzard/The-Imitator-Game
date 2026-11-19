# 2026-10-21 续研：统一首图入口、累计预算修复与隐藏物体反例

本轮从 v1 最终快照恢复，297个文件在开始时全部吻合；检查进程、任务锁与磁盘后继续本地工作。新增 API 调用0、GPU执行0、合成人类图像/视频/有效配对均0。已有仿真、视频、失败与 v1 快照未覆盖。完整集成 **724 passed，13 subtests passed，21.30秒**，见 [日志](runs/20261021_harness_v2/final_integration_pytest.log)。

产物根目录：`/mnt/sdg/datasets/real_sim_human_pipeline/harness_20261021_v2/`。旧同步多视角的物理成功和跨历史执行精确字节不一致结论继续保留；本轮没有重跑仿真或放宽该门。

## 先更正来源记录

v1报告及旧 CURRENT_STATE 把两布局描述为“同一真人来源”，这是记录错误。实际执行 input、receipt、contract 与媒体 hash 一致指向：

| 局部开发/验证角色 | 实际真人窗口 | 录制会话 |
|---|---|---|
| g00 development | `ego_part2_basic_pick_place_2174` | `2025-04-01_13-41-06.mov` |
| g01 validation probe | `ego_part2_basic_pick_place_6057` | `2025-04-01_13-54-42.mov` |

同一 contract 枚举多个可用来源，不能将 contract 相同误写成所选来源相同。新增 [source_lineage_audit.py](human_to_robot_pipeline/source_lineage_audit.py) 从实际 receipt 选择来源，检查对应关系、文件 hash、重复执行和跨数据划分泄漏。重新读取旧15次与新2次，共17份成功执行记录，得到两个真人窗口、两个注册表内会话、8组精确字节数据。旧15次仍精确复现相同的6组成员划分。新增2个字节组来自已记录的极小浮点差异，不声称新行为。

旧报告的 digest 序列化未明确，直接比较历史 digest 字符串未通过；本轮用明确的拼接字节规则复现的是组成员划分，不宣称 digest 字符串相等。历史记录保持原样。15份旧记录没有执行 UUID；两份同步记录有 UUID，统计中分别保留这一差别。

两个来源在 strict 历史记录里都已出现，**不是新来源盲测**，数据 split 均为 train。g01 只是候选冻结后、未用于本轮实现调试的接线探针。证据：[逐文件更正](/mnt/sdg/datasets/real_sim_human_pipeline/harness_20261021_v2/independent_review/lineage_correction.json)、[实际来源清单](/mnt/sdg/datasets/real_sim_human_pipeline/harness_20261021_v2/lineage_inventory/records_01.json)、[历史分组比较](/mnt/sdg/datasets/real_sim_human_pipeline/harness_20261021_v2/lineage_inventory/history_comparison.json)。旧状态保存在 `prior_current_state.json`，旧 v1 报告添加了有日期的勘误。

## 统一入口已接线，视觉收益未验证

此前人物计划和仿真状态包各自有效，但没有共同绑定所选真人窗口、实际物理执行和相机；旧人物计划的 lineage 是调用方文本。另一个可区分的问题是把全量状态差分拼到首图请求时，会把终态动作带进初态。单独增加单人提示不能修复这些接口缺口。

新 [unified_anchor_candidate.py](r2h_pipeline/unified_anchor_candidate.py) 是 opt-in 候选：

1. 从实际 receipt 的 structured proposal 追溯精确真人窗口、session、媒体、时间区间、fps、split 和人工语义审查；重新从原始数组编译所选相机状态，比较对象角色与帧内容。
2. 从共同包派生人物计划和任务角色；首图仅包含初态、唯一操作者及静止手角色。未来事件分工和终态差分留在视频阶段。
3. 图像执行已接入既有 Gemini provider。视频准备及提交已接入既有 Seedance provider，但必须先核对实际生成回执，再分别通过未改动的 `single_actor_v2.1` 与独立新规则 `static_state_v1.0`。

新静态门分别判断任务、对象身份、相对尺寸、角色、初态关系、非目标保持与相机对应。它是严格标注/证据校验器，不是自动视觉检测器；VLM全肯定且无直接观察仍为 unknown，静态图不能证明动作顺序、接触力或整段视频连续性。规则由独立审核分支冻结，生成分支未修改同轮标准。旧单人规则和历史负例未改变。

| 真实接线或负例 | 结果与范围 |
|---|---|
| g00开发、冻结后g01 | 两包均可重新验证；g00对面视角源在左，g01源在右；首图请求均未混入终态 |
| 来源/session/window/fps/执行/相机/角色/首终帧/verification变异 | 10/10拒绝，是真实源一致性反例，不是视觉检出率 |
| 保持actor-plan hash但换来源、重算包hash换相机、终图作首图、注入未来动作 | 5/5在生成前拒绝 |
| 真实execute-initial | 两包均被零预算拒绝，没有付费开始标记 |
| 真实video gate | 缺生成回执、人物审查或静态状态审查时阻断 |
| 历史人物回归重放 | 131个绑定复查；10份判断仍7 reject、3 needs_review，未重复运行无改动的边界检测 |

真实 provider 成功响应只在明确标记的单元测试替身里覆盖接线；不能将其计作成功生成。使用说明与输入输出在 [UNIFIED_ANCHOR_CANDIDATE.md](r2h_pipeline/UNIFIED_ANCHOR_CANDIDATE.md)。真实探针在 `unified_anchor/development/` 和 `unified_anchor/validation/`；独立反例在 `independent_review/source_mutations_final/` 与 `package_dispatch_negatives/`。

## 累计预算的实际实现漏洞

独立审计用无网络的本地 operation 复现了旧 `single_actor_candidate.py` 的问题：账本由预算文件路径派生，复制完全相同的配置可以获得另一份计数。没有真实 API 超支；旧路径只作冻结基线，不用于新付费实验。

新统一入口使用固定 campaign 注册表，把 campaign_id、canonical ledger 和限额绑定；相同配置副本共享同一账本，改变账本位置或提额会被拒绝。未知提交继续占额，异步视频在终态查询前仍占并发槽；同一阶段恢复读取已有结果。

独立9项预算检查通过，包括复制配置恢复/耗尽、换账本、提额、未知提交、并发冲突、零预算和异步任务。另用本地查询替身确认既有 poll 函数兼容 canonical origin。相关测试目录中的 call_started 是明确隔离的 fixture，**不是付费调用**；真实 unified campaign 的 marker 数为0。本轮 image/video/review累计额度全部为0、并发上限1。人民币历史账单仍未知，不从 token 推算。

## 隐藏物体：找到一个有效告警和一个明确漏检

复用既有 SIFT 注册、仿真投影、抽帧与ROI工具，新增 [hidden_identity_extent.py](r2h_pipeline/hidden_identity_extent.py)。本轮使用一个已有柜门来源、三条历史视频、每条三帧；9帧不是9个独立样本。

协议在数值比较前冻结：参考区域归一化后的宽/高增长下界超过2或面积增长下界超过4才给强比例告警，两边ROI边界传播±2像素误差。它是工程诊断区间，不是统计置信区间。目标ROI、对应柜格与可比视角仍由直接看图确认，记录9次人工/agent介入；源物体投影取自已绑定的仿真QA。

| 历史视频 | 相对固定柜体的高度/面积增长下界 | 诊断 |
|---|---|---|
| A原始 | 2.326 / 4.575 | 严重比例不一致 |
| A同请求重抽 | 1.834 / 2.885 | unknown；直接可见仍错误，保留为漏检 |
| B终态条件 | 0.690 / 0.513 | unknown；不授予身份通过 |

全局SIFT末帧注册有73/77/107个内点，但两个A在源小物体/柜格内部的内点均为0：匹配到背景不能证明隐藏内容身份。以柜格归一化还会掩盖物体和柜格一起变大的情况，因此同时保留固定柜体参照。相似颜色或尺寸都不构成身份通过。

6类输入变异全部保守unknown。根agent另外直接查看4张关键覆盖图，并独立重新解码10张唯一原视频帧、核对PTS与像素、重算9项区间；结果一致，保留A重抽漏检。没有正常新来源的误报率或自动检出率。证据：[方法与局限](r2h_pipeline/HIDDEN_IDENTITY_EXTENT.md)、[原诊断](/mnt/sdg/datasets/real_sim_human_pipeline/harness_20261021_v2/hidden_identity/run_01/report.json)、[独立复核](/mnt/sdg/datasets/real_sim_human_pipeline/harness_20261021_v2/hidden_identity/independent_audit_01.json)。

## 决策与下一轮

保留统一入口作为候选，优先替代旧路径开展后续受预算约束的资格实验；当前不晋级默认生成器。独立静态门可用于审查，但没有新合成图像通过它。尺寸诊断只保留为半自动反例工具，不提升为自动身份验收；不能靠换seed、提高全局注册分数或调低门槛抹掉漏检。

下一轮本地工作优先补**人物位置的几何可操作性和初态遮挡预检**：目前身体位置/进入区域由结构化文字指定，尚无身体姿态或可达性证据。随后将隐藏对象的实际显露状态、局部身份参考和相对尺寸接入条件准备，并继续保留对应不足时的unknown。预留杯盖等新来源须先具备对应仿真几何/抓取资格，不能直接当严格规划已泛化。

一旦配置新的明确累计API预算，最小视觉资格实验为g00单张首图，人物与初态对象分开审查；首图通过才尝试视频，再在冻结后使用独立来源验证。现在每条有效配对的时间/成本均为null，分母为0。没有策略训练，没有干预GPU1/3训练进程；结束时GPU2仍21 MiB，任务锁未占用。动态现场见 [CURRENT_STATE.json](../CURRENT_STATE.json)，长期规则仍在 [HARNESS.md](../HARNESS.md)。
