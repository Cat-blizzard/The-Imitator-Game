# 同步 TIG 官方 main

2026-11-17，合并官方 `3193957e3b0187d086e44d79e0e4af9845fbd4fd`，其相对开发基线 `d6d16ec511bc389e0a207692730c137bc022ef14` 增加 6 个提交、变更 88 个文件。保存成果的原始快照 `2a712a49d48256f7ddc2ab932502271b63f4358e` 保留在提交历史中。本次只更新 Draft PR 的功能分支，fork 的 `main` 及本地参考、运行、原开发 checkout 均保持原版本。

官方更新包含 Zero-WAM、Astra、ACT 默认配置、文档和忽略规则。官方没有修改 `mani_skill`、根 `pyproject.toml` 或 `examples/baselines/lerobot_dataset/video_utils.py`。合并冲突仅在根 README 和 `.gitignore`：保留官方全文和 `.omo/*`，加入本地文档入口与运行文件忽略规则。

75 个双臂模块、6 个 worker、原开发 README、3 个环境文件共 85 个文件仍符合原始 source manifest；视频寻帧修正保持原 blob。全部官方更新的文件中，除根 README 和 `.gitignore` 的本地补充外，其余内容与官方一致。历史协议、结果、source manifest 和 snapshot validation 没有重写。

同步后的检查包括上述源码与证据哈希核对、Python 语法、调度/错峰/任务适配器三项 CPU 自测，以及官方新增 Astra 的 8 项报告测试。具体命令、退出码和范围记录在 [upstream-sync-validation.json](upstream-sync-validation.json)。这些是同步兼容性与 CPU 检查，不代表重跑历史原生实验。

ACT 默认模型结构由 encoder/decoder 2/4 层、hidden 256、FFN 512、8 heads 增加为 12/12 层、hidden 1024、FFN 4096、16 heads。以后使用未显式覆盖参数的 ACT 命令时，应按对应 checkpoint/config 核对模型结构。该官方改动不改变当前双臂规划/控制接口。

此次没有重装 TIG/cuRobo 环境、下载模型或运行新基线训练/仿真。新加入的 Zero-WAM/Astra 完整运行与同步后的原生 GPU 回归均未验收。原有固定路径、解释器、GPU 选择、外置证据及资产定位仍需后续迁移适配；同步官方源码不会自动解决这些问题。
