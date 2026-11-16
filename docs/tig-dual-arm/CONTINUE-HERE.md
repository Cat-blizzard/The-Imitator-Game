# 新服务器继续工作的交接说明

2026-11-17。用户的目标是继续 `TIG-IMPLEMENTATION-PLAN.md` 中的仿真、数据生成与 Benchmark 基础设施 TODO。当前主线是 TIG 上的双臂公共执行器与 cuRobo，不接入独立 RoboTwin 项目，也暂不推进仍在其他地方训练的 BPP。

## 先确认恢复状态

1. 阅读同目录 `MIGRATION.md`，使用迁移包的 manifest 和 SHA256 校验、恢复原相对布局。
2. 确认参考 d6d16ec、视频修复运行树 27164a18、开发树为 manifest 中的提交、cuRobo 42498cf4 且 tag v0.9.0。开发树已合入官方 3193957。
3. 重建 tig-official 与 curobo-tig 两个环境。先做 CPU/API/资产与路径检查，再在目标机器上实际验证一个完整任务和动作回放。新机通过与原机通过分别记录。
4. 旧历史文件中保留原绝对路径；换 ROOT 后按迁移文档处理，禁止批量替换旧协议、源码归档和哈希。部分自测读取早期 fixture，完整 validation/dual-arm 已随包保留。

## 当前成果

- 共同时间轴采取“同时开始、各自完成、短轨迹终点保持”，统一执行器每周期同时提交两臂和夹爪动作。
- 已有组合、世界与持物碰撞检查、真实抓稳、滑移/掉落、释放/退手/末段稳定及动作/状态/视频时间关系检查。有限采样检查不代表连续碰撞证明。
- M1 三任务为苹果香蕉入篮、药瓶分区、水果同盘；完整源盘/支撑适配和早期配对结果见完整历史证据。
- cuRobo seed401 的三个独立冻结开发版本完整闭环和实际动作回放分别为 198、197、196 控制步；不可合并为同版本跨任务成功率。
- cuRobo 固定五个未见种子 1001–1005 的原计分为 3/5，MPLib 4/5。原生携物冲突夹具两后端均执行通过，但不替代 cuRobo 空间重规划、交接和失败恢复的独立验收。
- 水果 v4 完整体积腕部覆盖使两臂 link7 各 64 球，保留 v2 持物球 194/603。旧 v3 四规划器初始化 OOM、早期碰撞拒绝、药瓶 MPLib 实际碰桌均保留。
- 旧 MPLib 侧交接及恢复成果分别在其独立历史协议中；它们不能直接记成 cuRobo 后端已完成的能力。

## 恢复运行后优先做什么

1. 修复主任务 seed1003 的保守包含误拒，并保留原协议失败计分。原网格顶点事后诊断支持“保守外包框拒绝”，不支持把实体判为越界。
2. 处理 cuRobo seed1004 的收尾预算不足，保持原 200 控制步预算及完成边界、原碰撞/滑移/退手/稳定条件，不延长预算换成功。
3. 对 MPLib 原生螺旋路径加入实际末端位姿残差检查，针对药瓶直接释放路线的约 10 mm Z 残差另做新版本验证，保留旧物理失败。
4. 将三个任务冻结为同一新版本，用少量开发未用种子回归；通过后，再预先冻结每任务 20 个开发未用种子的 MPLib/cuRobo 配对，共 120 次。历史开发种子不能重新命名为未见种子，正式分配以审计过的使用记录为准。串行/同步是另一对照维度，分别计数。
5. 再分别推进 cuRobo 空间重规划、完整交接、失败恢复；持物质量/质心/惯量目前只导出审计，TimeOpt 动力学接入另列，不能声称已验收。

## 关键证据位置（恢复后的 ROOT 下）

- 总 TODO：`infrastructure/TIG-IMPLEMENTATION-PLAN.md`
- 当前跨任务与 cuRobo 结论：`infrastructure/validation/dual-arm/curobo-complete-20261116/CUROBO-COMPLETE.md`
- 最终审计：同目录 `completion-audit-item5-v2.json`
- 主五种子：同目录 `paired-unseen-protocol-v1.json`、`paired-unseen-summary-v1.json`
- 水果 v4：同目录 `fruit-link7-protocol-v4.json`、`fruit-link7-summary-v4.json`、`fruit-link7-v4-run-receipt.json`
- 历史 M1、交接、碰撞夹具及环境：完整 `infrastructure/validation/dual-arm/`
- 开发源码：`infrastructure/development/tig-dual-arm/mani_skill/examples/motionplanning/dual/`
- 6 个 worker：外层 `infrastructure/curobo_tig_*worker.py`

原始结果始终按当时的协议、源码和配置解释。后续修复与新机运行各自新增版本、输出目录和验收记录，保留全部成功与失败。
