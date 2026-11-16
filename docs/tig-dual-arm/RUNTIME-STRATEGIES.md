# 双臂运行去重与质量策略：开发检查点

**最新验收见[最终本地验收](INFRA-CONVERGENCE-ACCEPTANCE.md)。** 下文保留94发布检查点的状态，其中NOT_RUN及待办描述当时边界；入口与配置用法仍适用。后续`f390fdf`已完成代表任务、第三水果任务、可选配置、记录对照及回放验证。

2026-11-17。阶段 3 的去重改动已完成两例原生生成与严格动作回放。阶段 4 的任务配置、可选收尾和常规录制已接入开发代码，源码冻结于 `fce20a5`，当前按 CPU 集成检查点发布；新策略的原生生成和动作回放仍为 **NOT_RUN**。本页不代表五阶段收敛目标已完成。

机器可读证据见 [阶段 3 对照](evidence/infra-convergence-stage3-comparison-20261117.json)及[本次检查点](evidence/runtime-strategies-checkpoint-20261117.json)。历史报告、轨迹、协议和原评分保持原样。

## 已验收的去重改动

`fb8a425` 直接校验当前 worker 请求，移除运行链上的旧格式转换及完整请求复制；一次原生更新的最终状态读回由相关层共享。没有跨更新复用动态状态。`779d9d3` 将正常执行的完整组合轨迹检查交给执行器，规划诊断路径仍可独立检查；相同物理子步的接触结果在记录和监测之间复用。

场景、机器人基座、夹爪、实际持物关系及源支撑离开状态的更新和失效检查继续保留。静态签名检查的实测开销较小，本轮没有为此引入跨场景缓存。

在独立 checkout 的 `779d9d3` 上，药瓶 seed401 的 MPLib 和 cuRobo 原生生成、严格动作回放分别完成 171 和 191 步，官方、物理、质量、录制结果均通过。相对各自旧版本，同例动作及全部五个状态叶子逐项相同；这是两个案例的结构回归，不能外推为全任务成功率。

cuRobo 同例 12 次 update RPC 合计从 4.5523 秒降至 2.8613 秒；整进程从 206.2673 秒变为 205.9217 秒。每个版本只测了一次，初始化和缓存也影响总耗时，因此这里只报告更新开销变化，不宣称端到端提速。

## 已接入、待原生验证的新策略

- `runtime_profiles.py` 按任务选择并冻结现有 17 字段执行配置。`legacy-strict-v1` 不接受覆盖，保留旧数值；`task-default-v1` 提供显式任务覆盖，默认抓稳、滑移、退手和稳定窗口数值保持既有值，额外收敛余量默认改为 0。官方 200 步时限及最后完成步 199 不可覆盖。
- 预算依据所选阶段和实际所需观测计算。取消统一额外三步余量否决，同时继续记录无法容纳已选阶段的预算事件。预算不足和已发生物理失败分别报告。
- 抓稳和滑移配置传到执行、持物导出、worker 校验及回放。质量参考保持原抓取时关系；新策略下的碰撞预测按实际控制步刷新持物关系，并在提交下一条持物命令前检查当前快照。有限采样不构成连续碰撞证明。
- 退手和最终稳定窗口可单独关闭。关闭稳定质量项后仍须观测实际释放、目标区域及官方任务结果；可立即确认时不额外推进仿真，否则在剩余预算内继续观测。关闭质量项不会生成“质量通过”的观测证据。
- `normal` 录制保留全部动作、控制步、初态、T+1 状态、版本、配置及必要事件，不渲染视频，不归档完整子步和全量机器人接触。100 Hz 接触与抓取监测、完整内存观测及现有审计仍运行。`diagnostic` 保留详细记录；常规模式的内存占用和仍保留的审计成本尚未优化。
- 回放读取源轨迹冻结配置，不能用当前默认值重新解释旧数据。严格源码核对与显式跨版本回归继续分开；常规模式的任务关键状态核对与背景确定性诊断分开记录，真实接触监测仍包含背景物体。

源归档、配置、动作、初态和必要物理证据的完整性检查继续生效。质量、预算或录制结果不会覆盖环境实际的官方成功结果；没有证据的项目保持未评估。

回放另修正了稳定窗口重设参考帧时少计一帧的问题：与生成端相同，将满足基本条件的当前帧计作新窗口第1帧。区域、速度和漂移阈值不变，实际旧／新函数的 CPU 复现已通过；新原生动作回放仍待验证，不追溯修改旧报告。

## 入口与配置

环境、资产和 worker 路径设置沿用[独立 checkout 说明](PORTABLE-RUNTIME.md)。输出目录必须尚不存在。新的公开入口当前用于开发验收，默认选择 `task-default-v1` 和 `normal`：

```bash
"$TIG_WORKSPACE/infrastructure/envs/tig-official/bin/python" -B -m \
  mani_skill.examples.motionplanning.dual.generate \
  --backend mplib --task TwoRobotPickPillToRegions-v1 \
  --seed 401 --planner-seed 0 --approach-recipe standard-8cm \
  --payload-refinement fixed-task-v1 --transport-recipe direct-release-v2 \
  --quality-profile task-default-v1 --recording-mode normal \
  --output-dir /absolute/path/to/new-native-output

"$TIG_WORKSPACE/infrastructure/envs/tig-official/bin/python" -B -m \
  mani_skill.examples.motionplanning.dual.replay \
  --input-dir /absolute/path/to/new-native-output --replay-mode strict \
  --recording-mode normal --output-dir /absolute/path/to/new-replay-output
```

`--quality-overrides /absolute/path/to/overrides.json` 接受当前任务的显式 JSON 对象。例如 `{"require_retreat": false, "require_final_stability": false}` 关闭这两个可选质量要求；它不是已经验证的推荐采集配置。允许字段及单位由 `runtime_profiles.ALLOWED_OVERRIDES` 和 `RuntimePolicy` 定义。完整解析结果、覆盖值和哈希进入每份报告。

已有 `validate_curobo_complete` 入口继续默认使用 `legacy-strict-v1` 与 `diagnostic`，便于原协议验证。公开 `generate` 也可显式选择这两个旧设置。生成和回放使用同一执行与记录实现。

## 尚需完成的验收

1. 在最终冻结版本重跑现有四个代表案例及动作回放；此前阶段 2 的四例、阶段 3 的两例不代替新策略验证。
2. 建立第三个水果任务的同版本基线，验证可选阶段和修改后窗口的配置确实传到原生执行与回放。
3. 复用受影响负例，确认失抓、掉落、禁止接触、未释放及持物模型失效仍能检出，保留全部失败。
4. 固定一个案例，在同版本比较常规与诊断模式的必要事件、耗时、归档量及动作回放结果，报告实测值。
5. 完成新入口的独立 checkout 启动、当前 README 和最终目标逐项验收后，再决定是否结束草稿状态。
