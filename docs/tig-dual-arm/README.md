# TIG 双臂生成与动作回放

在 TIG 内置的 ManiSkill 代码上扩展双 Panda：分别规划两条臂轨迹，由同一执行器按共同时间轴提交动作，短轨迹结束后保持终点与夹爪状态。MPLib 与 cuRobo 共用执行器；当前默认链路不是14维双臂联合求解。

目前支持苹果香蕉入篮、药瓶分区、水果同盘三个任务。常用入口是 `dual.generate` 和 `dual.replay`；旧 `validate_*` 工具保留用于历史验收和诊断。

## 开始使用

先按[环境与路径说明](PORTABLE-RUNTIME.md#入口与路径用法)设置 `TIG_WORKSPACE`、`PYTHONPATH`、资产及单张渲染 GPU。需要现有 TIG 环境；cuRobo 另需独立环境和依赖。这不是一键安装包。

从当前 checkout 运行，输出目录应尚不存在：

```bash
"$TIG_WORKSPACE/infrastructure/envs/tig-official/bin/python" -B -m \
  mani_skill.examples.motionplanning.dual.generate \
  --task TwoRobotPickPillToRegions-v1 --backend mplib \
  --seed 401 --output-dir /absolute/path/to/new-generation

"$TIG_WORKSPACE/infrastructure/envs/tig-official/bin/python" -B -m \
  mani_skill.examples.motionplanning.dual.replay \
  --input-dir /absolute/path/to/new-generation \
  --output-dir /absolute/path/to/new-replay
```

切换规划器只需改为 `--backend curobo`。单次生成按任务和后端选择已有配方，显式高级参数优先：

| 任务 ID | 接近配方 | 持物表示 | 运输配方 |
| --- | --- | --- | --- |
| `TwoRobotPickAppleBananaToBaskets-v1` | compact-4cm | baseline | standard |
| `TwoRobotPickPillToRegions-v1` | MPLib: standard-8cm；cuRobo: compact-5cm | fixed-task-v1 | direct-release-v2 |
| `TwoRobotPickFruitsToPlate-v1` | standard-8cm | fixed-task-v4 | standard |

`--help` 显示常用用法，生成入口的 `--help-all` 显示高级选项。显式 manifest 批次沿用文件中的配置，不自动补上述任务配方；不会自动重采样失败种子。

## 默认行为与结果

- 生成使用版本化 `task-default-v1`：保持现有抓稳、滑移、退手和稳定窗口数值，取消统一额外三步余量。执行与回放共用冻结配置，官方200步限制保持。
- 退手与最终稳定窗是可选质量项。可通过 `--quality-overrides /absolute/path/to/task-quality.json` 选择当前任务的配置，例如 `{"require_retreat": false, "require_final_stability": false}`；关闭它们也会跳过相应可选阶段，仍检查实际释放、目标区域和官方结果。新配置单独计分，不能套回旧报告。
- 默认 `normal` 保存动作、初态、T+1状态、配置及必要事件；不保存视频和完整子步归档。接触、抓取等必要监测继续运行。逐阶段 Python 输出放在 `console.log`，终端显示简短结果。已有内存审计仍运行，不能据此宣称消除了全部审计开销。
- `--recording-mode diagnostic` 启用详细输出和录制。`report.json` 分开给出官方结果、执行、物理、质量、录制；规划拒绝和质量不足不等同于已发生物理失败。结果只涵盖已观测部分。
- 公开回放默认 `cross-version`，验证源归档完整性和动作兼容，记录当前版本差异，并按源轨迹配置执行原动作。需要相同运行源码时选择 `--replay-mode strict`。旧验证入口仍默认严格源码核对与诊断录制。

检查责任保持清楚：后端生成和求解；执行器检查完整组合轨迹并推进环境；运行监测检查实际接触与持物状态；质量配置选择接受标准；报告层汇总事实。正常执行不在 runner 再完整扫描同一轨迹。详见[检查与质量约定](CHECKS-AND-QUALITY.md)。

## 验证与范围

2026-11-17完成的30例固定配对批次：官方成功17/30、原配置合格16/30。药瓶两后端均5/5合格；其他任务的差异受MPLib端点IK拒绝及支撑几何门影响。这是修复前的完整实现对照，不能证明cuRobo总体更快或更好；旧结果和失败记录保留。

本轮已修复内部IK接受与最终FK要求不一致、螺旋修正停滞及两类支撑几何误拒，并简化上述公开入口。[修复与收敛验收](CONVERGENCE-REPAIR-20261117.md)保存已知案例回归、八条严格动作回放、两条跨版本回放及保留的失败。已知失败回归不能当作新的未见种子成功率。

历史能力与证据分别见：

- [五阶段基础设施验收](INFRA-CONVERGENCE-ACCEPTANCE.md)、[运行策略及配置](RUNTIME-STRATEGIES.md)。
- [早期修复验收](REPAIR-20261117.md)、[水果碰撞表示修复](FRUITS-COLLISION-COVER-REPAIR.md)。
- [同步数据到ACT](DATA-ACT-BRIDGE.md)：数据转换、训练、模型回到仿真已接通，首轮策略任务超时。
- [cuRobo跨回合复用实验](CUROBO-EPISODE-REUSE.md)：独立性能实验，不作为本轮默认路径或优势结论。
- [长期实施方案](IMPLEMENTATION-PLAN.md)、[原成果快照与版本历史](SNAPSHOT-AND-HISTORY.md)。

源码仓库不包含完整资产、环境、模型及原始H5/视频，也不代替数据备份。现有接触检查是100Hz离散观测；完整交接、通用恢复和14维联合规划需分别验收。
