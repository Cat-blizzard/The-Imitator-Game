# ReSimPair

真实人类任务窗口 → 任务与技能合同 → 机器人仿真轨迹 → 合成人类参考视频。

2026-10-31 最新[资产范围、尺寸与释放复盘](reports/asset_release_rethink_20261031/README.md)：935抓取来源不是全58,117库筛完；追溯出旧归一化继承和小物采样偏置。37个原尺寸容器CPU处理新增7候选，独立输入包可读取155容器来源，默认148不变。一个三物混合上方释放任务通过原独立验收；2场景/8个尺寸覆盖任务仅编译、未执行物理。详细状态和后续见 `START_HERE.md` / `CURRENT_STATE.json`。

2026-10-31 新增[场景观察、诊断与任务修复接口](docs/SCENE_TOOLS_V1.md)：`scripts/scene_tools.sh` 统一调用现有后端、版本化任务和累计预算执行器。真实两场景对照中，7物新落点版本通过，8物原目标重排仍失败；[结果与根因](reports/architecture_20261031/README.md)。新接口可用，控制器候选未整体替换默认版本。后续已追加第11个 `scene_apply_release` 工具，原10接口阶段的证据保持独立。

2026-10-27 最新：[一套脚本抓取不同资产并放到指定目标](docs/REQUESTED_PICKPLACE.md)。新增 `cache_runtime init --requests`，只读 cuDex 缓存，按物体尺寸、容器内腔和目标坐标编译场景，保留共同控制与独立物理验收。8 来源、17 请求的实际对照为原配置 13/17、接触参数候选 15/17；候选有新回归，未替换默认。[结果、失败及视频](validation/requested_pickplace_20261027/RESULTS.md)。

同日较早阶段：[64 资产三方案实际对照与可迁移抓姿缓存](validation/grasp_comparison_20261027/RESULTS.md)。GraspGenX / cuDex / 混合方案分别有 24 / 44 / 48 个来源通过共同视觉几何筛选；实际完整抓放 7/7、12/13、10/13，按预登记 26 来源计为 7/26、12/26、10/26。保留所有预检与执行失败，不外推全库。新增 [无优化器轨迹入口](docs/PRECOMPUTED_GRASP_RUNTIME.md)：克隆后使用仓库内或指定文件夹的 pose、原生模型与容器，无需 cuDex / GraspGenX；隐藏原工程、断网的实际复验 2/2 通过。

本仓库包含全部 64 来源的三方案抓姿研究归档，以及 17 个完成原生碰撞模型准备的资产包。新入口见 `scripts/setup_cached_poses.sh` 与 `resimpair_pickplace.cache_runtime`。完整上游 cuDex 仓库、权重和优化环境不上传；新增资产的离线求解仍为可选研究流程。

同日较早的候选更新：[实际 cuDex 原生抓放对照与自动多物体收集](validation/cudex_physical_20261026/RESULTS.md)。相同八资产的完整抓放为几何 7/8、cuDex 优化候选 8/8；多物体收集 2/4，笛卡尔抬升修复仅完成离线验证。该历史实验使用不同的种子数，与最新 64 来源试验分别统计。

这是现有研发管线的独立运行仓库，不是重新实现。主线代码、冻结的历史方法依赖、ManiSkill/PolicyGen 运行代码、机器人模型、家居资产、审核器、API 客户端和当前私有配置均在本目录。运行新实验不需要访问原来的 `zhiqi`、独立 `PolicyGen` 工程或旧实验目录。原工程和旧实验没有被迁移或覆盖。

**本仓库包含所有者要求保留的实际 API 凭据，只能上传到私有仓库。** 凭据位于 `private/.env.local`，没有加入 Git 忽略；模型和服务地址在 `configs/providers.json`。

2026-10-26 的新候选：[通用抓放、容器、宽布局与同场景多任务](docs/GENERAL_PICKPLACE.md)。使用 `scripts/run.sh pickplace`，保持原默认入口不变。累计 11,461 个资产含离线抓取候选；原生执行资格另验，不能把候选数当作成功数。[实际结果与失败](validation/general_pickplace_20261026/RESULTS.md)按开发、保留验证、方法对照分别记录。

## 安装

已验证环境：Linux x86_64、Python 3.12、NVIDIA GPU、CUDA 12.8 PyTorch wheel、SAPIEN 3.0.2。主机还需要兼容的 NVIDIA 驱动、Vulkan、`ffmpeg` / `ffprobe`。这些系统依赖不能用复制 Python 代码替代。安装会下载锁定的依赖；代码与内置资产不从旧工程读取。

```bash
cd ReSimPair
python3 -m pip install --user uv  # 若尚未安装 uv
bash scripts/setup.sh
bash scripts/run.sh doctor --gpu 2
```

`doctor` 检查依赖、619 个冻结方法文件的哈希、资产路径和 API 协议资格证据；不提交收费请求。GPU 编号应选择当前空闲的设备。执行器遇到已有 GPU 进程或占用时会停止，不结束别人的任务。

## 从同一场景生成多个任务

以下是两个不同任务共享一个初始场景的最小例子。首次执行请先跑小批，然后根据独立验收结果扩大。

```bash
bash scripts/run.sh init --name demo --gpu 2 \
  --kinds linear_faders --layouts 1 --tasks-per-scene 2 \
  --image-limit 1 --video-limit 2 --review-limit 6 --usd-limit 40
bash scripts/run.sh native --root runs/demo
bash scripts/run.sh audit --root runs/demo
bash scripts/run.sh export --root runs/demo \
  --audit runs/demo/audits/native.json --output runs/demo/dataset
```

`init` 只准备场景、任务、预算、验收器绑定及冻结代码。`native` 才启动物理执行。默认 `smoke` 阶段每种机制只执行首个布局的前两个任务。已完成或已保留的尝试不会重新提交。

主线参数化机制包括 `linear_faders`、`rotary_indexer`、`spring_console`、`rocker_console`、`hook_retrieval`、`granule_sweeping`；接触扩展包括 `support_gate_release` 和 `lever_payload_lift`。CLI 会按现有枚举校验名字。`--layouts` 为 1–5，`--tasks-per-scene` 为 1–4；某些机制只提供三个任务。保留失败，不保证每个组合通过。

更多布局在资格验收后运行：

```bash
bash scripts/run.sh native --root runs/demo --phase base \
  --audit runs/demo/audits/native.json
```

接触扩展使用原管线的 `contacts` 阶段；不合格机制不能通过扩大采样绕过资格检查。

## 生成人类参考视频

这是收费步骤，只从独立验收合格的仿真中选择。每个场景先做一个共同人物首图和两个不同任务的视频；人物、对象、动作审核未通过时停止扩大。

```bash
bash scripts/run.sh human-prepare --root runs/demo \
  --audit runs/demo/audits/native.json --max-scenes 1
bash scripts/run.sh human --root runs/demo --phase qualification
bash scripts/run.sh status --root runs/demo
```

实际调用使用配置中的 Nano Banana、Seedance 和 Gemini。请求、响应、视频任务 ID、审核、失败、计费记录保存在 `runs/demo/`，不把原始密钥写进提示词或报告。调用并发固定为 1，HTTP 超时 600 秒，单视频轮询期限 900 秒，不自动重抽。未知提交必须先恢复查询，不能换目录重新提交。

`--usd-limit` 是服务额度换算的**名义美元准入上限**，并非已经核对的人民币账单。调用次数、返回额度和已观测费用分别记录。每次新调用前还有保守额度预留检查；有调用次数余量也可能因额度不足而停止。新预算继承原历史的 29 次图片、22 次视频、50 次审核提交，加上仓库迁移验证账本及后续本地调用，不能清零旧消耗。`runs/` 与 `runtime/` 是运行账本，备份时必须一起保留；不要用新克隆绕过已有提交。

## 接入自己的真实视频

批量机制示例默认没有真人来源，明确标记 `synthetic_only`，不能当作已确认的真人—机器人配对。真实输入是运行数据，应放在仓库内，例如 `inputs/raw/clip.mp4`：

```bash
bash scripts/run.sh pipeline --root runs/demo -- intake-source \
  --source source_01 --video inputs/raw/clip.mp4 \
  --session session_01 --split train --start 0 --end 120
bash scripts/run.sh pipeline --root runs/demo -- planning-input \
  --source source_01 --execute
bash scripts/run.sh pipeline --root runs/demo -- interface
```

窗口索引采用原生视频帧，使用 `[start, end)` 半开区间。来源抽取使用已登记的审核调用预算。模型给出可见对象、角色、事件、技能与不确定性，经过结构和证据绑定检查后交给规划接口。

**任意新任务仍需要满足已有技能和规划接口。** 从可见视频自动得到任意可执行任务，尚不是本系统已验证的能力。适配新的机制后，用原入口的 `intake-case` / `register-planning` 注册场景、任务或规划 bundle；保留源视频证据、事件协议和前后条件。可用 `pipeline --root runs/demo -- --help` 查看完整参数。旧的按日期重放命令保留为研究代码，使用历史实验数据的重放不属于新仓库的最小运行路径。

## 文件布局

| 路径 | 用途 |
|---|---|
| `resimpair.py`、`resimpair_runtime.py` | 独立 CLI、相对路径、资产迁移、预算继承 |
| `corl2026_workshop/experiments/autonomous_pipeline_v4/` | 当前任务生成、仿真、多视角、合同、图像/视频生成及审核 |
| `corl2026_workshop/experiments/task_contract_v3/`、`state_anchor_v2/` | 当前主线仍使用的底层模块 |
| `corl2026_workshop/tools/` | 独立物理验收和证据审核工具 |
| `data_generation_methods/` | 619 个逐文件保留的冻结依赖，含严格规划、R2H、资产选择和客户端 |
| `PolicyGen/mani_skill/`、`PolicyGen/examples/baselines/` | 仿真运行库、控制器、机器人资源及必要辅助模块 |
| `assets/household/` | 12 类家居资产目录及原始网格；不等于 12 种已验证任务机制 |
| `assets/precomputed_grasps_v1/` | 可直接用于原生轨迹的相对路径 pose、网格与容器包；17 个来源 |
| `assets/precomputed_grasp_research_v1/` | 64 个来源的三方案原始 pose 与视觉网格检查归档 |
| `resimpair_benchmark/`、`resimpair_pickplace/cache_runtime.py` | 有限三方案对照、计时与读取预计算抓姿生成新任务 |
| `private/.env.local`、`configs/providers.json` | 实际凭据与服务/模型配置 |
| `requirements.lock`、`scripts/` | 锁定依赖、安装、启动与发布工具 |
| `provenance/`、`validation/` | 来源哈希、历史消耗、协议证据及迁移验证结果 |
| `runs/`、`runtime/` | 当前实验、媒体、动作状态、账本、资产重定位索引；本地运行生成 |

`.venv`、运行产物和缓存不上传 Git；实际 API 配置按所有者要求上传到私有仓库。历史全量视频、训练集和既有运行目录没有打包；它们不是启动新实验的代码依赖。第三方代码沿用随附许可证，本次没有变更其授权。

## 验证与边界

最新的[批量桌面姿态与抓取库](docs/BULK_TABLETOP_POSES.md)通过 `scripts/run.sh tabletop` 使用。12,000 个原始网格离线计算产生 5,767 个带抓取候选的资产、225,870 个 pose；该层不运行逐物体动力学仿真。资产的尺寸、支撑姿态、接触点与失败原因保留，按需转换给现有 `grasp` / `tasks` 执行器。全量缓存与小样本物理结果分别统计，见[本轮结果](validation/tabletop_bulk_20261026/RESULTS.md)。新候选没有替换旧默认，也没有把几何候选全部标成物理成功。

2026-10-26 后续新增[全库角色检索、按需加工和同场景多任务工厂](docs/ASSET_LIBRARY_AND_TASK_FACTORY.md)。通过 `scripts/run.sh assets` / `scripts/run.sh tasks` 使用，候选场景库从 28 个加工描述扩为 258 个；静态、操作几何、抓取候选、物理通过和人类视频分别计数。[运行与失败证据](validation/asset_scaleup_20261026/RESULTS.md)保留原始版本，不替换默认管线。

2026-10-26 新增可选的[资产整理与通用网格抓取入口](docs/GENERIC_GRASP_PIPELINE.md)：本地资产角色筛选、抓取缓存、随机布局和多类干扰物、严格路径检查、同步双视角及现有人类视频执行器预检。结果见[实际对照记录](validation/generic_grasp_20261026/RESULTS.md)。默认机制保留，候选资格不等于全库操作资格。

```bash
.venv/bin/python -B -m pytest -q
```

具体检查和真实失败见 [validation/RESULTS.md](validation/RESULTS.md)。迁移验收隐藏了原 `/mnt/sdf`、`/mnt/sdg` 工程，只暴露这个仓库，另在不同挂载路径启动新实验。仿真通过、视频生成成功、视觉审核合格、正式配对可用分别统计，不合并成一个“成功率”。

已有实验目录中的绝对路径和内容哈希属于证据，不会为了迁移静默改写。**代码换路径后可新建实验；正在运行或已冻结的实验不能直接搬家后续跑。** 迁移这些历史运行需要另行做可核对的证据映射。API 服务、资产下载和驱动仍是外部运行条件；未安装资产的额外机器人模型不自动获得任务适配。

## 发布到指定 GitHub 账号

已发布：[fufumeichi/ReSimPair](https://github.com/fufumeichi/ReSimPair)，可见性为 **private**。首次发布的 5,886 个文件逐一核对了远端 Git 对象哈希，包含 `private/.env.local`；记录见 `validation/github_publication.json`。

仓库本地提交身份为 `fufumeichi <1991722272@qq.com>`。模型 API key 不具备 GitHub 建仓权限。先在终端完成一次账号登录，再运行发布脚本：

```bash
gh auth login --hostname github.com --web
bash scripts/publish_private.sh
```

脚本确认登录用户为 `fufumeichi`、目标仓库为 private 后才推送；不强制覆盖已有分支，不打印凭据。

2026-10-26 后续的[夹持、支撑与释放调查](docs/GRASP_RELEASE_STABILITY.md)已完成候选实现和小样验证：实体手指闭合检查、朝向支撑预检、实际负载更新、分阶段释放，以及可选快速 CPU 碰撞转换。详见[结果](validation/grasp_release_20261026/RESULTS.md)。支架夹持倾斜改善，但原鱼缸底面仍不合格；保留验证 2/2，不能外推全库。共享物理额度现为 24/24，旧验收、失败和默认主线保留。

2026-10-26 最新的 [cuDex-graspvla 调查与批量瓶颈分析](docs/CUDEX_GRASP_QUALITY_AND_BATCHING.md) 已完成：精确核对 Franka 延长指／DFC 配置；8 个来源做离线接触搜索与静力代理对照，6 个产生候选。新代理排除了一个既有物理成功来源，故仅保留研究候选，未替换默认。新增只读批次耗时与结果恢复工具；本轮物理、GPU、收费 API 均为 0，详细证据见 [结果](validation/cudex_contact_quality_20261026/RESULTS.md)。
