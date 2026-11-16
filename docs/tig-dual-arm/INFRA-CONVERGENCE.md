# 双臂基础设施收敛：阶段 1 开发快照

2026-11-17。本次保存现有双 Panda、MPLib/cuRobo 生成与动作回放的配置、结果和版本兼容改动。原严格配置已集中；相关 CPU 检查与四份旧录制的兼容性预检通过。**新代码的原生物理运行、实际动作回放与独立 checkout 启动尚未验证，阶段 1 验收仍未完成。**

工程取舍见 [CHECKS-AND-QUALITY.md](CHECKS-AND-QUALITY.md)，长期任务见 [IMPLEMENTATION-PLAN.md](IMPLEMENTATION-PLAN.md)。历史轨迹、协议和原评分保留；本次没有更换规划后端或引入新机器人。

## 当前实现

- `runtime_policy.py` 提供冻结的 `RuntimePolicy`、完整序列化、规范 SHA256 和报告恢复。当前入口使用 `legacy-strict-v1`：5 步抓稳、5 毫米位置滑移、0.15 弧度旋转滑移、8 厘米实际退手、20 步最终稳定；200 步官方时限和原 3 步收敛预算拒绝继续生效。执行、物理滑移审计和回放读取同一份配置。尚未开放任务配置 CLI，初始数值保持原协议，原生行为一致性待回归确认。
- `run_results.py` 为新生成与回放报告添加 `results`，分别保存 `official`、`execution`、`physics`、`quality`、`recording`、`events` 和 `acceptance`。官方结果来自环境观测；未观测为 `null` / `unknown`。首次成功、最终成功、时限和实际截断分开记录。`acceptance` 保留原完整运行的接纳决定与退出码含义。
- 录制或视频错误仍可令数据接纳失败，但不会改写已观测的官方成功或已有闭环结果。损坏的 ticks 文件保留原样，汇总记录录制错误，再保存其余可用终态。
- 实际目标接触、双臂接触和完整机器人／环境接触分别计数。确认发生的禁止接触会记为物理失败；缺少完整监测证据时保持未知。代理几何拒绝、质量滑移和预算事件分别记录，不据此推断实际碰撞或失抓。物理覆盖仍是 100Hz 离散观测，不是连续碰撞证明。

## 回放模式

现有回放入口增加 `--replay-mode strict|cross-version`，默认 `strict`。

| 模式 | 源归档完整性 | 当前执行依赖源码 | 其他要求 |
| --- | --- | --- | --- |
| `strict` | 核验完整清单与文件哈希 | 保留原六项关键依赖一致性检查；新配置录制增加 `runtime_policy.py` | 原动作、状态、控制频率、关节映射及资产兼容检查继续生效 |
| `cross-version` | 相同核验 | 允许差异，记录源版本、当前文件身份及差异 | 相同的动作与兼容性要求；真实回放仍须另行执行并通过 |

新回放读取源录制的配置；缺配置字段的旧录制使用原严格配置。新增的资产兼容核对涵盖任务角色、原生 actor/state 身份、碰撞顶点和初始角色位姿。跨版本模式不改写旧归档，不重新规划，也不逐帧纠正状态。

## 已完成的验证

[阶段 1 验证摘要](evidence/infra-convergence-stage1-20261117.json) 保存受测文件 SHA256、命令、退出码与证据身份。

- 7 个模块的 CPU 自测实际退出码均为 0：配置、统一执行器、滑移审计、结果汇总、录制收尾、完整生成入口和回放入口。覆盖录制失败不覆盖官方成功、缺失观察、200 步边界、完整机器人碰撞分类、初始滑移与实际失抓的区分，以及损坏 ticks 文件。
- 四份旧录制（192、187、191、171 动作）在新代码下通过 `cross-version` 的 CPU 预检；`strict` 均正确拒绝已变化的关键执行依赖。旧输入及源档哈希保持不变。**预检未创建仿真环境，不能视为实际动作回放通过。**
- 修改运行代码前，补跑并保存第三个水果任务 seed 401 的同版本基线：cuRobo 原生 196 步、严格动作回放 196 步，退出码均为 0。这是迁移前基线，不能代替当前新代码回归。

CPU 自测可在已配置的 TIG 环境中运行：

```bash
python -m mani_skill.examples.motionplanning.dual.runtime_policy --self-test
python -m mani_skill.examples.motionplanning.dual.dual_arm_execution --self-test
python -m mani_skill.examples.motionplanning.dual.validate_sync --self-test
python -m mani_skill.examples.motionplanning.dual.run_results --self-test
python -m mani_skill.examples.motionplanning.dual.multitask_recording --self-test
python -m mani_skill.examples.motionplanning.dual.validate_curobo_complete --self-test
python -m mani_skill.examples.motionplanning.dual.validate_curobo_complete_replay --self-test
```

大型轨迹、视频、资产、虚拟环境和完整本地日志不随源码上传；发布摘要注明外置证据及原件哈希。

## 后续阶段与完成条件

| 阶段 | 后续工作 | 完成条件 |
| --- | --- | --- |
| 1 配置、结果与回放 | 对冻结的新代码跑原生生成、严格回放及旧轨迹跨版本实际回放 | 配置贯穿三端，观察结果分层正确，原接纳协议无意外变化 |
| 2 调用链与路径 | 提取运行依赖的 `validate_*` 公共函数，统一 worker 启动与请求校验，分开本仓库代码与外部依赖 | 从独立 checkout 实际启动，消除对旧开发目录结构的依赖 |
| 3 重复开销 | 根据调用关系和实测耗时去重，保留规划、组合检查及物理监测的不同职责 | 状态变化后正确刷新或失效，不重复同阶段的完整扫描与读回 |
| 4 新策略 | 按任务配置质量、可选退手和稳定窗口，取消统一 3 步余量否决，区分常规与诊断归档 | 新配置另存版本；实际时限、失抓、掉落和禁止接触监测继续生效 |
| 5 回归与交付 | 四个代表案例与动作回放、第三任务、受影响负例、常规／诊断实测比较及文档 | 有清楚的生成与回放入口、独立启动证据，成功与失败均保留 |

阶段 2 候选尚未接入，本次没有交付轻量记录模式或新质量接纳策略。后续每阶段分别提交、记录源码和证据身份；不追溯重计历史结果。
