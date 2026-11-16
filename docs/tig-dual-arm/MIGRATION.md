---
license: other
pretty_name: TIG TODO workspace transfer
---

# 在新服务器继续 TIG TODO

这份私有迁移包保存当前双臂规划与控制、cuRobo 接入、实验记录及仿真资产，用于恢复开发工作区。代码源于 Cat-blizzard/The-Imitator-Game 的 Draft PR #1，已同步官方 main 至 `3193957`；包内 `manifest.json` 给出实际保存的开发提交。原始成果快照、官方参考、视频修复运行树和 cuRobo 的提交身份分别保留。

## 包含内容

| 文件 | 恢复内容 |
| --- | --- |
| `code/tig.bundle` | TIG Git 历史、同步后的开发分支和旧参考；恢复参考、开发、运行三个 checkout |
| `code/curobo.bundle` | cuRobo 固定源码 `42498cf4` 及 `v0.9.0` tag，保留原生 worker 所检查的 Git 身份 |
| `workspace-support.tar.zst` | 外层 6 个 worker、TODO、安装/检查入口、环境与下载身份记录 |
| `dual-arm-evidence.tar.zst` | 完整双臂验证目录；包括早期 fixture、M1 配对、交接、cuRobo，保留开发/正式/成功/失败及历史源码 |
| `simulation-assets.tar.zst` | 当前完整 `infrastructure/maniskill`：YCB、任务物体、背景、PartNet、Sketchfab 等 |
| `environment/` | 两个 Python 环境的当前安装快照、可移植依赖输入和旧环境配置原件 |
| `inventory/` | 所选源文件元数据清单，用于核对范围与数量 |
| `manifest.json`、`SHA256SUMS` | 数据包组件、大小、源码提交和完整文件 SHA256 |
| `restore_workspace.py` | 仅校验、解包、恢复 Git 布局与新机器环境入口；不安装依赖或运行 GPU |

整理前的文件内容统计：双臂验证目录约 **18.20 GiB**，仿真资产约 **25.25 GiB**；归档排除 `__pycache__` 与 `.pyc`。以 manifest 中的实际数量和体积为准。新服务器需同时容纳下载包、解压后的约 44 GiB 数据、源码及后续重建的环境；环境大小另外预留。

未随包搬迁：整套 TIG 官方训练数据、独立 BPP/RoboTwin/Zero-WAM 项目、模型权重、旧虚拟环境、下载/编译缓存、重复资产压缩包、派生的本机绝对路径 URDF、HF 登录信息。现有仿真开发可使用包内资产继续；后续人类参考、训练/基线评测、新本体与软体工作按相应 TODO 另补数据和依赖。官方数据下载脚本及来源清单随 support 包保留。

## 新服务器下载与恢复

准备 Linux、Python 3.12、Git 和 `zstd`。HF 客户端只负责下载，恢复工具自身只用 Python 标准库。私有仓库需要在新服务器单独登录具有读取权限的身份；登录凭据不在迁移包里。

```bash
hf auth login
hf download Duoliu080106/TIG_infra --repo-type dataset --local-dir ./tig-transfer

python3 ./tig-transfer/restore_workspace.py verify --package ./tig-transfer
python3 ./tig-transfer/restore_workspace.py restore \
  --package ./tig-transfer --destination /your/new/ZeroWAM-for-TIG
```

`--destination` 必须不存在或为空，工具拒绝覆盖已有工作区。完整归档校验与成员检查会读取所有压缩包，较小文件的数量较多，需要等待。恢复流式解压，不额外生成一份几十 GiB 的未压缩 tar。也可以直接 rsync 整个迁移包目录到新服务器，再执行同一恢复流程。

恢复后的布局：

```text
ROOT/
  The-Imitator-Game/                         # d6d16ec，历史 clean 参考
  infrastructure/
    development/tig-dual-arm/                # manifest 中的开发提交
    runtime/The-Imitator-Game/               # 27164a18，视频修正
    backends/curobo/                         # 42498cf4，v0.9.0，clean
    maniskill/data/                          # 仿真资产
    validation/dual-arm/                     # 完整历史证据
    migration-environment/                  # 安装快照与旧 env 原件
    common.env                              # 为新 ROOT 生成
    tig-development.env                     # 为新 ROOT 生成
    curobo_tig_*worker.py
```

TIG 的三个 checkout 共享 Git 历史，origin 指向 Cat-blizzard fork；cuRobo origin 指向 NVlabs 官方仓库。不能只把 PR clone 成单层目录来代替此布局，当前代码使用固定的相对层级定位 worker 和 fixture。原参考固定旧提交是为了保留已验证接口和安装器检查；后续开发使用已经同步官方的 development 树。

## 重建两个环境

旧 venv 的 Python 是指向原机安装目录的绝对链接，不随包迁移。下面是基于原机安装快照的重建命令；它们尚未在新服务器验收，需要网络下载包。原机为 Linux x86_64、Python 3.12.14。

```bash
TIG_RESTORE_ROOT=/your/new/ZeroWAM-for-TIG
source "$TIG_RESTORE_ROOT/infrastructure/common.env"

uv venv --python 3.12.14 "$TIG_RESTORE_ROOT/infrastructure/envs/tig-official"
uv pip sync \
  --python "$TIG_RESTORE_ROOT/infrastructure/envs/tig-official/bin/python" \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  --index-strategy unsafe-best-match \
  "$TIG_RESTORE_ROOT/infrastructure/migration-environment/tig-official.portable-requirements.txt"

uv venv --python 3.12.14 "$TIG_RESTORE_ROOT/infrastructure/envs/curobo-tig"
uv pip sync \
  --python "$TIG_RESTORE_ROOT/infrastructure/envs/curobo-tig/bin/python" \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  --index-strategy unsafe-best-match \
  "$TIG_RESTORE_ROOT/infrastructure/migration-environment/curobo-tig.portable-requirements.txt"
uv pip install \
  --python "$TIG_RESTORE_ROOT/infrastructure/envs/curobo-tig/bin/python" \
  --no-deps --no-build-isolation "$TIG_RESTORE_ROOT/infrastructure/backends/curobo"
```

`*.raw-freeze.txt` 是原机记录。cuRobo 的本地 `file://` 安装项仅从 portable 输入中移除，再从固定源码安装；没有改用 PyPI 最新版。保留 `v0.9.0` tag 是为了 setuptools-scm 的版本推导。TIG 的 torchcodec 官方 wheel URL 保留。安装快照是当前实装清单，不冒称经过新机器验证的完整环境锁。

原机关键版本：Torch 2.7.1+cu128、NumPy 1.26.4、SAPIEN 3.0.3、MPLib 0.1.1、gymnasium 0.29.1；cuRobo 发行名为 `nvidia-curobo==0.9.0`，Warp 1.19.0。原机使用 Ubuntu 22.04.5、NVIDIA 驱动 580.178.04、CUDA Toolkit 12.8、约 48 GiB 显存的 RTX 5880 Ada；这些是观测配置，不代表已测得的最低需求。旧水果四规划器配置曾采样到接近 46 GiB 显存占用，新 GPU 需要单独核对资源。

旧 `install-tig-official.sh` 校验参考必须为 d6d16ec，且部分完整验证读取未随迁的训练数据/权重，不能用它的完整结果替代本次迁移验收。TIG 原机严格依赖检查仍有 7 项来自官方 override 的声明冲突，历史记录保留；cuRobo 环境原检查通过。

## 路径及验收

恢复器生成的新 `common.env` 指向新 ROOT，开发入口为：

```bash
source "$TIG_RESTORE_ROOT/infrastructure/tig-development.env"
CUDA_VISIBLE_DEVICES='' python -m mani_skill.examples.motionplanning.dual.trajectory_schedule --self-test
CUDA_VISIBLE_DEVICES='' python -m mani_skill.examples.motionplanning.dual.conflict_coordination --self-test
CUDA_VISIBLE_DEVICES='' python -m mani_skill.examples.motionplanning.dual.multitask_adapters --self-test

CUDA_VISIBLE_DEVICES='' "$TIG_RESTORE_ROOT/infrastructure/envs/curobo-tig/bin/python" -B \
  "$TIG_RESTORE_ROOT/infrastructure/curobo_tig_worker.py" --self-test --check-native-api
```

上述命令是新机验收步骤，不表示它们已在目标服务器执行。先做零 GPU 验证，再挑空闲 GPU 跑一个完整任务与实际动作回放，将新机结果另存，不覆盖旧记录。GPU/驱动、Vulkan 渲染和 CUDA 构建工具均需在目标机器实际检查。

还有三处路径相关事项：

1. **旧绝对路径保持原样。** 历史协议、日志、源码归档和审计里仍记录 `/data/jxc/ld/ZeroWAM-for-TIG`；部分旧审计会按这些路径查文件。当前回放入口的 CPU 自测也有一个旧 ROOT 字面量断言。若新机能恢复完全相同的真实路径，可直接满足这些定位；选择其他 ROOT 时，需要后续有记录的路径适配或兼容挂载。恢复器没有改写旧证据，也没有把这些旧审计标成在新 ROOT 通过。
2. **派生 URDF 重新生成。** 包内只有原始机器人模型和生成代码；旧 `tig-derived-models` 带原机 mesh 绝对路径，不作为新机运行输入。
3. **背景随机化路径。** 当前三个默认任务使用包内默认纹理；官方随机背景逻辑另读 `~/.maniskill/data/robotwin/background_texture`。开启时需核对该路径与新资产根的映射，保留已有用户资产，避免覆盖。

## 继续推进的 TODO

原始完整方案位于 `infrastructure/TIG-IMPLEMENTATION-PLAN.md`。当前优先处理 cuRobo 未见种子的保守包含拒绝、收尾预算和 MPLib 原生端点残差；三个任务同版本小批回归通过后，再冻结新的正式配对。cuRobo 的空间重规划、交接、失败恢复分别验收。现有三任务开发成功与原五种子 3/5 计分保持各自版本身份。

## 许可证和访问

迁移包按组件保留原许可：TIG 仓库许可证、cuRobo 的 NVIDIA/Apache 2.0 声明、第三方资产各自的许可与来源。包的 `license: other` 表示组件按各自许可使用，不为第三方资产重新授予统一许可。仓库用于私有迁移，原始证据中的本机路径、GPU 标识和日志原样保留。包不包含 HF token、Git 认证配置或完整用户环境变量。
