# HyperICLZQ 10k 训练入口

数据：`/mnt/sdf/datasets/hypericlzq_curated_10k_v2/`。先查看这里的 `REPORT.md`、`summary.json`、`external_task_overlap.json` 和 `gallery.html`。`manifest.json` 的 action-space schema 是动作约定的唯一来源。

这一版允许外部任务与原 unseen 任务重合；原 benchmark unseen 和 held-out 轨迹不进入训练。测试指标仍使用原 seen/unseen/L0–L3，但解释为外部数据迁移，不能统一称为完全未见任务的零样本学习。

## 新动作约定

- Panda 和 AgiBot 都是 `[L Δq7, L open, R Δq7, R open]`，`open=0` 关，`open=1` 开。整段目标相对同一个当前状态，不能连续累加。
- Panda 监督是原控制指令；AgiBot 监督是未来实测关节运动，权重 0.5；并未验证它是可直接执行的电机命令。
- 人类为当前相机系下的双手 `Δxyz3 + ΔR6 + pinch距离1`，20 维，权重 0.35。米、关节弧度、旋转矩阵列的语义没有混用；人类 pinch 没有伪装成机器人开合百分比。
- 三种本体有各自的归一化和动作适配器，共享 32 维动作隐空间、DiT 和 hypernet。没有新增未校准的 FK/IK、机器人之间的关节逐项等价假设。
- `evaluate.py` 支持 v1 绝对关节输出和 v2 相对关节输出；v2 在进入原仿真控制器与时间聚合前恢复为绝对目标。

## 先验证，再训练

在 PolicyGen 根目录运行：

```bash
.venv/bin/python -m unittest \
  examples.baselines.hypericlzq.tests.test_training \
  examples.baselines.hypericlzq.tests.test_batching \
  examples.baselines.hypericlzq.tests.test_action_representation -v
```

正式训练建议先用单卡，保证数据/输出/评估链路可追溯。下面命令只是准备好的入口，本次数据整理不会执行正式训练：

```bash
.venv/bin/python -m examples.baselines.hypericlzq.train \
  --manifest /mnt/sdf/datasets/hypericlzq_curated_10k_v2/manifest.json \
  --output runs/hypericlzq_10k_v2 \
  --steps 20000 --batch-size 8 --accumulate 2 --num-workers 4 \
  --lr 0.0001 --warmup 500 --aux-warmup 2000 \
  --source-ratios imitator_sim=.50,imitator_human=.05,agibot=.25,egodex_pose=.15,egodex_video=.025,epic=.025 \
  --future-weight .05 --order-weight .02 --task-weight .02 --consistency-weight .02
```

训练 loss：本体有效维度上的 flow-matching 动作误差、同精确目标双参考行为一致性、参考视频任务对比、因果未来 DINO 特征预测和时序方向判断。原生动作监督通过生成的参数回传 hypernet decoder；纯视频预测主要训练编码器与上下文模块，不会凭空得到动作标签。

原任务仍占 50% 动作抽样，不按 10k 的原始条数直接均匀混合。对外部动作标签的置信权重与来源抽样比例分开控制。跨数据源弱目标配对的动作 loss 再乘 0.5–0.7；弱配对不会被强制赋予精确同任务的一致性目标。

先做小规模端到端 rollout，检查机器人能否执行相对动作还原后的目标；损失下降不能代替成功率。若原任务性能下降，先将原仿真比例提高到 0.6，降低外部人类动作权重/频率，检查参考消融及标签时间对齐，再考虑增大模型。

## 评估与对照

使用训练产出的实际 checkpoint 文件路径：

```bash
.venv/bin/python -m examples.baselines.hypericlzq.evaluate \
  --checkpoint runs/hypericlzq_10k_v2/last.pt \
  --suite unseen --output runs/hypericlzq_10k_v2/eval_unseen \
  --num-episodes 10 --reference-mode correct
```

训练器保存 `last.pt`；评估前将目标 checkpoint 固定下来，避免后续训练覆盖。分别跑 seen、seen_heldout、unseen 和 correct/wrong_task/blank 三种参考条件。保持原 L0–L3 成功率指标，并结合 `external_task_overlap.json` 披露每个 unseen 任务的外部覆盖。

三个对照 manifest 已放在同一数据目录，架构一致，Panda 归一化一致：

1. `manifest_anchor_only.json`：原 3,000 仿真 + 750 人类参考，使用 v2 动作表示。
2. `manifest_external_video_only.json`：新增数据仅作视频/任务参考辅助，不使用其动作标签。
3. `manifest.json`：完整动作与视频混训。

固定每组步数、有效 batch、随机种子和评估配置。来源比例在缺少来源时会重归一化，做严格算力与原任务曝光量对照时需要显式设置各组比例。v1 checkpoint 的动作 schema 与 v2 不兼容，不能直接 `--resume`；当前初始化入口也会拒绝混用不同 schema/归一化。

数据筛选复现参数在各候选目录 `batch_report.json`；10k 合并参数在 `build_args.json`。扩量无需重复扫描通过/失败候选，已建立的视频偏移索引和数值检查结果会复用。

训练采样按“来源 → 任务类别 → 具体目标 → 轨迹”分层。抓放因目标位置不同有数百个 goal，不会因此淹没只有少量 goal 的清洁任务；每个来源内的任务类别保持平衡。
