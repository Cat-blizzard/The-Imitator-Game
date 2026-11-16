# TIG 双臂执行与 cuRobo：成果保存快照

当前开发进度见[运行去重与质量策略检查点](RUNTIME-STRATEGIES.md)：阶段 3 的去重已通过 MPLib／cuRobo 药瓶两例原生生成和严格动作回放，动作与状态保持不变；阶段 4 的任务配置、可选收尾、常规／诊断录制和公开生成／回放入口已接入，CPU 检查通过，最终新策略原生回归仍为 **NOT_RUN**。阶段 1、2 的证据分别见[发布快照](INFRA-CONVERGENCE.md)和[独立 checkout 检查点](PORTABLE-RUNTIME.md)。后续遵循[检查分层与质量配置约定](CHECKS-AND-QUALITY.md)，旧轨迹与评分保留。下文为历史快照。

2026-11-17 后续修复：快照 PR 已合入 fork `main`（`f65ebc6`）。基于该版本完成放置几何、MPLib 终点与收尾预算修复；同一源码的四例代表运行及实际动作回放均通过，见 [修复与验收说明](REPAIR-20261117.md)。下文保留前次快照保存时的状态与历史结果。

2026-11-17。此分支已保存开发成果，并合入官方 main 至 `3193957`，作为 Cat-blizzard fork 的 Draft PR 继续整理。当前扩展源码来自已验证工作区；同步后的 CPU 回归已通过，原生 GPU 运行和新机器复现尚未验证，路径与安装适配仍待完成。

## 保存内容

- `mani_skill/examples/motionplanning/dual/`：75 个 Python 模块及原开发 README，包含共同时间轴、统一执行、组合与持物碰撞检查、多任务适配、错峰、空间重规划、交接、恢复和录制回放。能力的验收范围分别以历史报告为准。
- `infrastructure/curobo_tig*_worker.py`：6 个独立 worker，包含进程隔离、持物配置、状态更新缓存、任务与规划策略。cuRobo 使用已有共同执行器。
- 3 个 TIG 官方环境输入与锁定文件、历史实施方案、cuRobo 阶段报告、成功及失败的协议/汇总副本和证据索引。
- 继承已有视频寻帧修正 `27164a1` 和调度提交 `1428b68`。本次保存没有改动原开发、运行或参考 checkout 的文件。

[source-manifest.json](source-manifest.json) 对 85 个原样复制文件记录 SHA256：75 个 dual 模块、dual README、6 个 worker、3 个环境文件。历史源码中的本机路径及 GPU 标识原样保留，以维持源码身份；这些设置需要在迁移时处理。

大型数据集、资产、虚拟环境、cuRobo 第三方源码、模型、H5/NPZ/视频和缓存没有上传。原始运行文件仍在来源机器，**PR 不是完整数据备份**。新增源码随本仓库现有许可证发布，上游版权与许可证保持。

## 版本关系

| 对象 | 本次核对版本 |
| --- | --- |
| fork `main` 与本地参考源码 | `d6d16ec511bc389e0a207692730c137bc022ef14` |
| 当前开发 checkout 的已提交 HEAD | `1428b682417b5eac70df5ccbec2ac0e219ca11e9`，另有本次保存的未提交开发文件 |
| 官方 `main`（2026-11-17 核对） | `3193957e3b0187d086e44d79e0e4af9845fbd4fd` |
| 当前 PR 分支 | 已合入官方 `3193957`；保留原始快照 `2a712a4` 及扩展源码 |
| cuRobo 已验证源码 | `42498cf4e7d0d6fbd4ef9fbcec1243c00c888a83`，版本 0.9.0 |

fork 的 `main` 与本地参考仍停在旧版；**当前 PR 分支已合入官方新增的全部 6 个提交**。差异集中在 baseline、README 和 `.gitignore`，官方 `mani_skill`、根 `pyproject.toml` 和视频读取文件与旧基线相同。同步保留扩展源码及视频寻帧修正，未改现有环境、原始开发/运行目录或旧实验结果；尚未在同步分支重跑原生 GPU 实验。官方比较：[d6d16ec → 3193957](https://github.com/imitator-game/The-Imitator-Game/compare/d6d16ec511bc389e0a207692730c137bc022ef14...3193957e3b0187d086e44d79e0e4af9845fbd4fd)。同步说明与验收见 [UPSTREAM-SYNC.md](UPSTREAM-SYNC.md)。

## 已有实际证据

以下是来源工作区已完成的运行，**不是本次发布重新运行的结果**。不同开发版本的源码及配置应分别按原协议识别。

| 开发验收 | 已有结果 | 解释边界 |
| --- | --- | --- |
| 苹果/香蕉 seed 401 | cuRobo 198/200 步；198 动作实际回放通过 | `compact-4cm`；不同 GPU 的 MPLib 184 步只作开发参考 |
| 药瓶分区 seed 401 | cuRobo 197 步及实际动作回放通过 | `standard-8cm`、`fixed-task-v1`、`pill-direct-release-v1`；同策略 MPLib 131 步实际碰桌失败保留 |
| 水果同盘 seed 401，v4 | cuRobo 196 步，MPLib 181 步；cuRobo 196 动作实际回放通过 | 同初态、配置及 GPU 的开发配对；完整进程墙钟 435.77 / 60.49 秒，不支持 cuRobo 总体更快 |
| 冻结未见种子 1001–1005 | cuRobo 3/5，MPLib 4/5 | 全部失败计入；事后诊断未把 seed 1003 重计成功；这是小批验证 |
| 原生携物冲突夹具 | cuRobo 176 步、MPLib 164 步通过 | 同步冲突候选先被拒绝，再执行错峰；回程由去程时间反转，未证明完整交接或恢复的 cuRobo 验收 |

水果 v3 的四规划器初始化 OOM、v1/v2 的原生拒绝、药瓶 MPLib 物理失败及未见种子失败都保留在协议与汇总中。主任务的成功示例没有替代未见种子正式计分，也不能合并成三个任务的总体成功率。

100Hz 接触观测是物理子步的离散检查，不是连续碰撞证明。持物质量、质心和惯量目前只导出审计，没有注入 TimeOpt 动力学。GPU 显存数值属于采样观测，不代表实测整程峰值。

详细阶段说明见 [CUROBO-COMPLETE.md](CUROBO-COMPLETE.md)，长期 TODO 见 [IMPLEMENTATION-PLAN.md](IMPLEMENTATION-PLAN.md)。二者为历史文档，部分相对链接指向外置的原工作区证据。

## 证据与源码身份

[evidence-index.json](evidence-index.json) 区分原文件 SHA256 与发布副本 SHA256，并注明仅索引、尚未上传的报告。`evidence/` 内 JSON 为脱敏派生副本：替换工作区、用户目录前缀、GPU UUID 和原始进程元数据，保留结果、协议、失败原因及历史源码哈希。不能用派生副本的文件哈希替代原始文件身份，也不能据副本单独重新验证外置轨迹。

`${TIG_WORKSPACE}` 指原外层项目，包含 `infrastructure/development/tig-dual-arm`、运行环境和验证记录；它不是此 Git checkout 的同义词。协议中的旧源码 SHA 对应各自归档。当前文件与某份旧协议不同是开发版本演进，不能拿当前源码替代旧归档宣称复现。

[historical-task-profiles.json](historical-task-profiles.json) 保存三个成功开发配置的线索。复现历史结果还需对应源码归档、资产、初态及协议；直接使用当前入口默认值不等于复现这三次成功。

## 环境与资产

已验证 TIG 侧为 Python 3.12、NumPy 1.26.4、gymnasium 0.29.1、SAPIEN 3.0.3、MPLib 0.1.1、Torch 2.7.1+cu128。cuRobo 侧为 Python 3.12.14、Torch 2.7.1+cu128、cuRobo 0.9.0、Warp 1.19.0，使用独立进程和环境。

`infrastructure/requirements-tig-official.in`、`.lock.txt` 和 `constraints-tig-official-overrides.txt` 保存 TIG 依赖输入。后者是原有 uv override 说明，不应直接当作普通 pip constraints；这些文件不是 cuRobo 环境锁。尚未交付经过新机器验证的 cuRobo 安装脚本或完整环境锁。

资产需要 TIG 的 YCB 与 `data/robotwin/objects`，包括药瓶、盘和水果；这是 TIG 任务自身使用的对象资源，不需要迁入整个 RoboTwin 项目。另有原代码直接读取 `~/.maniskill/data/robotwin/background_texture`，迁移需核对。Panda 原始 URDF/SRDF/mesh 已在 TIG 源码中；生成的 cuRobo URDF 带本机 mesh 路径，应在目标机器重新生成。第三方资产与 cuRobo 均按各自原始来源及许可准备。

## 草稿后续工作

- [x] 同步官方 6 个提交，保留双方 README/忽略规则及原始快照；源码身份与 CPU 回归通过。
- [ ] 将固定的 `parents[]`、旧嵌套 DEV 路径、本机解释器、GPU 选择和证据路径改为显式配置。不要复制一套重复的 TIG 来伪造旧目录结构。
- [ ] 提供 cuRobo 独立环境安装、干净固定源码、worker 启动和派生 URDF 生成流程，核对对象与纹理资产。
- [ ] 从历史证据抽取最小自测 fixture，建立可定位的外置证据包；为原始 H5/视频另做备份。
- [ ] 在新 checkout 实际执行 CPU 和原生回归，再将 PR 转为可合并状态。
- [ ] 继续解决未见种子的保守包含拒绝、收尾预算和 MPLib 末端残差问题；三个任务的新版本回归后另冻结正式配对。空间重规划、交接、失败恢复的 cuRobo 能力分别验收。

原始快照的发布检查保留在 [snapshot-validation.json](snapshot-validation.json)，其中 `upstream_sync=NOT_DONE` 描述保存快照时的事实；后续同步验收另存 [upstream-sync-validation.json](upstream-sync-validation.json)。原生 GPU 回归和新机器复现仍为未运行。
