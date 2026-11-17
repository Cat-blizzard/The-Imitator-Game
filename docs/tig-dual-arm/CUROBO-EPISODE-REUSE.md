# cuRobo 跨回合复用与配置构造诊断

2026-11-17。运行源码冻结于 `edcb153`。本次保留左右臂各 approach／transport 共四个 planner，增加兼容回合间的复用和完整重置，减少重复构造与预热；求解参数、碰撞要求、任务质量配置及 normal 模式的必要实时监测保持原设置。单回合 `generate`／旧验证入口仍沿用原生命周期，只有显式 manifest 批次进入新流程。

**首轮验收完成：14／14 项 CPU 自测、7 个正例生成、1 个实际未释放负控及 2 次严格动作回放均得到预期结果。** 同 GPU 冷启动与连续 A→B→A 各 3／3 通过，批次耗时从 621.33 秒降至 273.01 秒，合格轨迹产量从 17.38 提升至 39.56 条／小时。首轮实测只覆盖药瓶 seed401／402 开发案例，不代表全任务成功率。机器可读结果及原始文件哈希见[首轮验收索引](evidence/curobo-episode-reuse-20261117.json)。

**扩展回归已完成：三个任务各三个固定种子的冷启动／复用配对、三个未释放负控及恢复、25 回合连续运行和四次严格动作回放。** 九组配对的合格结果与动作均一致，观察到的整批吞吐提升为 1.70–2.29 倍；水果 seed402 的原生规划失败在两种生命周期下均保留。详见下文“三任务与 25 回合扩展回归”和[扩展证据索引](evidence/curobo-episode-reuse-extended-20261117.json)。

## 批次用法

环境、资产、单 GPU UUID 和解释器设置沿用[独立 checkout 说明](PORTABLE-RUNTIME.md)，质量及录制配置见[运行策略](RUNTIME-STRATEGIES.md)。以下 manifest 使用现有药瓶开发配方；`worker_script` 需替换为当前 checkout 的绝对路径。它也可以省略，以使用原有默认 task worker。

```json
{
  "schema": "tig-generation-batch-v1",
  "defaults": {
    "backend": "curobo",
    "task": "TwoRobotPickPillToRegions-v1",
    "planner_seed": 0,
    "approach_recipe": "compact-5cm",
    "payload_refinement": "fixed-task-v1",
    "transport_recipe": "direct-release-v2",
    "quality_profile": "task-default-v1",
    "recording_mode": "normal",
    "worker_script": "/absolute/path/to/checkout/infrastructure/curobo_tig_seed_policy_worker.py"
  },
  "episodes": [
    {"id": "A-cold", "seed": 401},
    {"id": "B", "seed": 402},
    {"id": "A-after-B", "seed": 401}
  ]
}
```

保存为 `/absolute/path/to/aba.json`，用公开生成入口分别比较两种生命周期。两个输出目录必须尚不存在：

```bash
"$TIG_WORKSPACE/infrastructure/envs/tig-official/bin/python" -B -m \
  mani_skill.examples.motionplanning.dual.generate \
  --manifest /absolute/path/to/aba.json \
  --output-dir /absolute/path/to/new-batch-output \
  --worker-lifetime batch

"$TIG_WORKSPACE/infrastructure/envs/tig-official/bin/python" -B -m \
  mani_skill.examples.motionplanning.dual.generate \
  --manifest /absolute/path/to/aba.json \
  --output-dir /absolute/path/to/new-episode-output \
  --worker-lifetime episode
```

| 生命周期 | worker 与四套 planner |
| --- | --- |
| `batch`，默认 | 首回合构造、预热；兼容后续回合执行重置，批次结束关闭 |
| `episode` | 每回合重新构造、预热、重置，回合结束关闭 |

两组的首个冷回合也调用相同的 `begin_episode`，保证比较使用相同重置协议。两组继续共享原有磁盘缓存；这里的“冷”指重新建立进程与 planner，不清空编译、Warp、CUDA 或 Torch 缓存。MPLib 仍逐回合运行，不使用 cuRobo pool。

manifest 使用单回合 parser 的参数名，例如 `planner_seed`、`approach_recipe`。每回合可覆盖 defaults；未写质量与录制设置时分别使用 `task-default-v1`、`normal`。相对文件路径以 manifest 所在目录解析。批次先检查全部参数、配置及案例 ID，再创建输出或 worker。ID 必须唯一、以字母或数字开头，后续仅允许字母、数字、点、下划线和横线，总长不超过 96；输出目录和 worker 生命周期由批次入口指定，不能由案例覆盖。

输出包括 `manifest.json`、`episodes/<id>/report.json` 和逐例更新的 `summary.json`。每例只尝试一次，失败保留并继续下一例，不自动重跑。汇总中的合格轨迹沿用既有完成协议，并要求录制通过；每小时合格轨迹数计入初始化、回合重置、失败及最终 worker 关闭时间。官方成功另按 `final_success_before_horizon` 报告，并分别给出全部案例、已观测案例的成功比例和未知数。

## 复用边界与重置

`CuroboEpisodePool` 至多持有一个 worker 进程，不按任务累积无限缓存。兼容性包含任务、GPU、worker 源码、planner seed、插值 dt、机器人与完整世界几何、持物容量及固定执行配置；不兼容时先关闭旧 worker，再显式构造新的四实例。场景及源支撑位姿可以在回合边界更新，几何拓扑、尺寸和惯性等变化需要重建。

`seed` 是环境初态种子，可以逐回合改变；`planner_seed` 是原生构造种子，同一组常驻 planner 内固定。当前 cuRobo 的 `MotionPlanner.reset_seed()` 恢复构造时的采样流，不接受任意新种子。更换 planner seed 或插值 dt 时，上层 pool 重建；底层 `begin_episode` 直接拒绝这类变更。重置 Python／Torch 随机状态和原生采样流不构成逐点轨迹一致性保证。

每次 `begin_episode` 接收当前实测、空手的双臂初态，完成以下工作：

- 四实例先一并失效，检查全部兼容性，再恢复场景位姿、手指锁定／默认状态、空手碰撞层与掩码，清除附着和源支撑许可。
- 更新当前 payload／source 模型，清理 grasp 历史、支撑撤销记录、snapshot tokens、版本及每回合计数；恢复原生采样状态。四个 planner 本身不重建，也不再次预热。
- 全部重置成功后，仍需下一次实测 `update_pair` 签发新 snapshot 才能查询或规划。相同 phase 名使用不同回合命名空间；报告只截取当前回合请求，冷回合保留四次初始化记录。

任一实例重置失败，全部实例保持不可用，不能用旧 token 继续。下一次显式完整重置可恢复；通信损坏则由 pool 更换 worker。失败案例的结果和证据保留，后续恢复不追溯改判。normal 仍保留禁止接触、实际抓取／失抓、释放、持物有效性和任务时限等既有运行检查。

## 配置构造诊断

默认仅记录原有初始化分段耗时。需要定位 `MotionPlannerCfg.create` 内部重复调用时，显式设置：

```bash
export TIG_CUROBO_PROFILE_CONFIG=1
```

沿用上面的生成命令并选择新的输出目录。四次初始化结果中的 `initialization.config_profile` 记录函数调用次数、self／cumulative 时间，以及 URDF、运动学、IK、TrajOpt、RRT、TimeOpt 等已识别入口。它只诊断配置构造，不改变传入参数，不共享可变原生对象，也不增加 CUDA 同步。累计时间有重叠，不能相加；factory 调用多次也不等于重复解析，应对照实际 URDF／运动学构造条目。

`TIG_CUROBO_PROFILE_CONFIG=1` 引入 profiling 开销，结果标记 `performance_baseline_valid=false`。性能对照时应 `unset TIG_CUROBO_PROFILE_CONFIG`，诊断批次单独保存；CPU profiling 也不能代替 GPU kernel 分析。本轮先采集证据，没有据此缓存原生配置或合并四实例。

独立四实例初始化诊断已完成：配置构造合计 113.46 秒，对象构造 1.38 秒，预热 40.05 秒（均含诊断开销）。各实例配置构造的 98.5%–99.7% 落在机器人碰撞模型准备链路；URDF 实际解析约 2 毫秒／实例。已确认官方精确自碰球对准备使用 CPU Python 双循环，当前左右臂分别有 2891／2627 个碰撞球。每实例内部 IK、TrajOpt、RRT 已共享运动学参数，主要重复发生在不同 planner 和不同回合。

后三次配置构造还观察到 20／20／10 秒线程等待；当前记录没有保留原始 caller edges，尚不能将其归因为 JIT 或某一种锁。下一步冷启动优化应针对不可变碰撞球对拓扑的构造，先核对可变掩码与持物更新的边界；不能直接复用整份带不同限位和动态状态的 RobotCfg。详细函数计数与分析见外置 `config-profile/analysis.json` 和 `config-profile/ANALYSIS.md`。

## 首轮验证证据

CPU 记录位于 `${TIG_WORKSPACE}/infrastructure/validation/dual-arm/curobo-episode-reuse-20261117/cpu-tests.json`：14 个模块入口均退出 0，源码前后哈希一致，未启动 GPU 仿真。覆盖四实例身份、三任务 A→B→A、源位姿刷新、部分重置失败后的全局失效及恢复、固定 planner seed／dt、回合请求切片、manifest 预检、两种生命周期及回放源码身份兼容。

同一张 NVIDIA RTX 5880 Ada Generation GPU（48 GB）顺序运行两组公开 manifest 批次，使用同一源码、配置、种子和现有磁盘缓存，profiling 关闭。表中批次执行区间包括初始化、重置、录制及最终关闭；完整 CLI 进程墙钟另为 625.50／276.46 秒，包含模块导入与预检：

| 生命周期 | A401、B402、A401 步数 | 合格／官方成功 | 整批秒数 | 合格轨迹／小时 |
| --- | --- | --- | ---: | ---: |
| 每回合新建四实例 | 191、183、191 | 3／3；3／3 | 621.33 | 17.38 |
| 一个 worker 跨回合复用四实例 | 191、183、191 | 3／3；3／3 | 273.01 | 39.56 |

该三回合批次吞吐为原来的 2.28 倍，墙钟减少 56.1%。连续组首回合 206.99 秒，后续 B、A 分别 28.42／34.69 秒，后两回合没有初始化 RPC，只有约 1.08 秒完整重置。冷启动组三回合各自初始化 RPC 为 168.55／162.74／157.03 秒。正常生成、全部检查和所选质量要求继续执行。

A→B→A 三回合保持相同四个 planner，初始化 RPC 数为 `[4, 0, 0]`；每回合均完整重置一次。A 初态恢复、B 初态确实不同；同案例冷暖动作以及返回 A 的 191×16 动作数组恰好完全一致，此一致性仅作为观测，没有新增为质量门槛。8 个回合、每回合 4 个 session 的重置读回全部通过已有世界位姿／层／enable／碰撞掩码、锁指／默认位置和空 payload 核对。全部原始报告的归档与严格源码身份通过。

失败序列另存 `failure-reset`：外置负控在 26 个 release 控制步把右夹爪 +1 改为 -1，未改臂命令或质量数值；176 步实际报告 `release_timeout: both_hands_not_open_or_grasp_persists`。官方成功仍为 true，但物理、质量与接受结果为 failed，合格轨迹为 false。批次原样保留 failed／退出码 1；验证汇总的 passed 表示预期负例被检出，不是把它改判成功。结束时原生 transport sessions 仍有 attachment，下一回合同一四实例完整重置后 A401 在 191 步通过，用时 31.73 秒。包含这次有意失败及冷启动的两回合序列为 239.98 秒、1 条合格轨迹；它与上表两组三回合性能计分分开保存。

复用后 `A-after-B` 和失败后 `A-after-failure` 的严格 normal 动作回放均退出 0、14／14 检查通过。各为 191 动作／192 状态帧，进程耗时 18.53／16.19 秒，只恢复初态一次，逐步状态修正与重新规划均为 false。官方、物理、质量和录制结果均通过；运行源码与回放输入前后哈希一致。

原始协议、各回合报告、初始化／重置读回及实验过程记录位于 `${TIG_WORKSPACE}/infrastructure/validation/dual-arm/curobo-episode-reuse-20261117`。`cold-aba/summary.json`、`continuous-aba/summary.json` 和 `failure-reset/summary.json` 保留各自原始计分；`comparison.json` 保存状态隔离、原生读回、动作比较和失败恢复分析；`replay-processes.json` 保存两次回放命令、退出码与哈希。仓库中的验收 JSON 是摘要和索引，大型 H5／NPZ 与环境资产仍在外置目录。

## 三任务与 25 回合扩展回归

扩展运行使用 `e7c519d` checkout，100 个运行源码文件与首轮冻结版本一致，未修改生产代码、规划参数、质量要求或官方 200 步时限。先冻结 `extended-regression/protocol.json`，再运行全部案例；每例只尝试一次，失败计入原始汇总。三个任务均使用环境 seed401／402／403、planner seed0、`task-default-v1` 和 `normal`，配方如下：

| 任务 | approach | payload refinement | transport |
| --- | --- | --- | --- |
| `TwoRobotPickAppleBananaToBaskets-v1` | `compact-4cm` | `baseline` | `standard` |
| `TwoRobotPickPillToRegions-v1` | `compact-5cm` | `fixed-task-v1` | `direct-release-v2` |
| `TwoRobotPickFruitsToPlate-v1` | `standard-8cm` | `fixed-task-v4` | `standard` |

每个任务的冷启动与复用批次在同一张 RTX 5880 Ada Generation 48 GB GPU 上顺序执行，profiling 关闭。不同任务及长序列在不同 GPU 上并行，共享宿主机 CPU 和缓存；下面的耗时用于描述本次运行，尚未隔离共享资源竞争。批次时间包含构造、重置、失败、录制和关闭。

| 任务 | 冷启动／复用合格数 | 冷启动／复用秒数 | 冷启动／复用合格轨迹／小时 | 吞吐比 |
| --- | --- | ---: | ---: | ---: |
| 苹果／香蕉 | 3／3；3／3 | 590.30／309.55 | 18.30／34.89 | 1.91× |
| 药瓶 | 3／3；3／3 | 588.57／257.48 | 18.35／41.94 | 2.29× |
| 水果同盘 | 2／3；2／3 | 907.64／533.65 | 7.93／13.49 | 1.70× |

九组冷暖配对的参数、初态及动作数组均一致，合格比例也相同。因此本轮支持复用提高生成效率，没有观察到轨迹质量提升。动作逐点相同仍只是一项观测，不新增为运行门槛。水果 seed402 在两种模式下都于第 137 步的 `carry_lower` 阶段报告 `CONSTRAINT_OR_POSE_FAILURE`；已有物理检查通过，质量结果未知，未计为合格轨迹，也未将规划失败解释为实际碰撞。

后续[seed402 原因诊断](FRUITS-S402-DIAGNOSIS.md)已用单例原生复现及失败候选几何复核定位到左臂 link5 对右臂 link6 的碰撞代理误拒。该诊断阶段未修改运行代码，本轮失败保持原样。后续双向link5／link6覆盖修复及独立回归见[修复验收](FRUITS-COLLISION-COVER-REPAIR.md)。

### 连续运行、失败恢复与回放

长序列为苹果 8 回合→药瓶 8 回合→水果 8 回合→返回苹果 1 回合。同一任务内复用四实例，切换任务时显式重建；worker generation 与协议一致，为 `[1×8, 2×8, 3×8, 4]`。25 回合中 23 条合格，用时 2297.41 秒，计 36.04 条／小时。两次失败均为水果 seed402 的同一规划错误，后续回合继续成功。原始批次保留 `failed` 状态和退出码 1。

每个任务另运行一个实际未释放负控，再在同一四实例上重置并运行正例。负控只将右夹爪释放命令从 +1 改为 -1，分别实际注入 20／26／22 步；三个负控均在 176 步被 `release_timeout: both_hands_not_open_or_grasp_persists` 检出。它们的官方成功为 true，物理、质量及接受结果为 failed，均不计合格；三个重置后的正例分别在 198／191／196 步通过。

三个任务的复用 seed401 及长序列返回苹果，共四条轨迹完成实际严格 normal 动作回放，各 14／14 项既有检查通过；动作／状态帧数分别为 198／199、191／192、196／197、198／199。均只恢复初态一次，没有重新规划或逐步状态修正，官方、物理、质量和录制结果均通过。

扩展实验共 49 个原生回合：42 条合格、3 个预设未释放负例、4 次水果 seed402 规划失败；四次回放另计。49／49 个回合通过既有重置读回与源码归档身份核对，未发生 OOM、重置失败或 worker 通信错误。

### 资源观测与证据位置

长序列在每回合边界采集设备显存及 client／worker RSS。下表是设备已用显存，不是峰值：

| worker generation／任务 | 回合数 | 设备显存范围（MiB） |
| --- | ---: | ---: |
| 1／苹果 | 8 | 39911–39919 |
| 2／药瓶 | 8 | 43411–43419 |
| 3／水果 | 8 | 45149–45211 |
| 4／返回苹果 | 1 | 39911 |

返回苹果后设备显存回到首次苹果观测值；各 worker 内 RSS 变化较小。client RSS 从首回合末的 1550260 KiB 增至末回合的 1961424 KiB，约增加 402 MiB，任务加载、缓存及其他保留内存的贡献尚未分离。本轮没有据此增加内存硬门槛；25 回合及边界采样不足以证明长期无泄漏。

完整记录位于 `${TIG_WORKSPACE}/infrastructure/validation/dual-arm/curobo-episode-reuse-20261117/extended-regression`。`protocol.json` 保存冻结计划，`comparison.json` 保存全部结果与配对／重置／源码分析，`*/observations.json` 保存资源观测，各 `summary.json` 与 `report.json` 保留原始成败。外置 `trial_driver.py` 只连接既有批次入口、注入已声明的负控并采集观测，没有替换验收规则。[扩展证据索引](evidence/curobo-episode-reuse-extended-20261117.json) 保存汇总、资源范围及原始文件 SHA256；大型轨迹仍在外置目录。
