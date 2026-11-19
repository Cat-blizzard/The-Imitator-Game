> 当前扩量入口：见 [TRAINING_10K.md](TRAINING_10K.md)。v2 使用相对动作约定，允许外部任务覆盖原 unseen 目标，但保留原评估轨迹隔离。下文 v1 的严格任务语义隔离与绝对动作说明仅适用于 v1 清单。

# HyperICLZQ：面向任务泛化的混合本体训练

这是独立训练入口，保留 `hypericl/` 原代码。当前阶段是数据筛选、实现和链路测试；没有正式训练结果。不要把单元测试通过理解为成功率提升。

## 架构与取舍

```text
参考视频 -> 冻结 DINOv2-L -> E -> T -> R -> D -> 共享 DiT 的 LoRA 增量
                                         ↑
                                 目标动作空间 embedding
当前图像 -> 同一个冻结 DINO -> 可训练 patch attention ─┐
当前状态 + 可见性 mask + 动作时间间隔 + 目标动作空间 ─┴-> policy 条件
本体专用动作输入适配层 -> 共享 Flow-Matching DiT -> 本体专用速度输出适配层
```

- E/T/R/D 沿用原实现的主要结构。残差生成器使用随机 A、零 B，避免双零初始化失去梯度。
- 原始仿真动作、人类估计手部运动、AgiBot 未来关节运动各有独立的输入/输出适配层。共享策略骨干及任务表征；不假设各动作坐标等价，也不做未经验证的手到机器人 retargeting。
- 当前观测也使用冻结 DINO，而非从零训练的视觉 CNN。阶段权重由当前观测与参考事件 token 的相似度计算，仍然不是有语义标注的进度百分比。
- 任务 ID 只用于参考配对和训练对比损失，部署模型不接收任务 ID。目标本体 ID 与参考视频的 human/robot 域分别传入。
- 参考优先来自同目标不同 episode；仿真 query 保留约 70% 概率选择对应的人类参考，其余可选择不同 L 等级机器人参考。外部数据只在自身可确认的目标组内配对，不把“都是放置”当作同一个目标。
- 外部 query 默认 20% 使用自身完整视频作为参考，其当前帧和动作交给 policy；其余优先同目标其他示范。可用 `--self-reference-prob` 调整。原仿真人类参考比例不受此开关影响。整段自参考可用于学习动作，但必须以不同示范参考的评测结果检验泛化。
- `R` 是学习出来的技能状态修正；当前版本不启用世界模型想象反馈、Latent TTT、任务专家库或参数蒸馏。先验证清晰的动作监督和参考依赖，再考虑这些复杂模块。

默认损失：动作 FM=1，未来特征=.05，视频顺序=.02，目标对比=.02，同目标双参考函数一致性=.02。可选 `--rgb-weight .01` 启用 32×32 未来 RGB 预测。辅助项用前 2000 步渐增，RGB 默认关闭以便首轮判断表征预测的增益。

未来目标随机选取 0.2、0.5、1、2 秒后的帧（受片段长度限制），上下文绝不包含目标或其之后的帧。日志同时报告 `future_copy_baseline`：直接复制最后一帧冻结特征的误差。预测误差需低于该基线，才能说明优于静态复制；这仍不能替代任务成功率。

动作 loss 回传到动作适配层、共享 policy、参数基、D/R/T/E 和当前观测投影；未来特征/顺序 loss 回传 E/T/R 与辅助头，不直接优化动作 D。无动作视频不会被填成零动作标签。对比损失不把不同来源中可能等价的目标强行当负例。同目标函数一致性用相同 query、噪声和 FM 时间，第二个参考追踪第一个参考的停止梯度速度场。

## 数据与质量口径

数据发布根目录：`/mnt/sdf/datasets/hypericlzq_curated_v1/`。原视频保留在既有公共数据目录；manifest 存档案成员/偏移或原视频路径，动作数组和精选短视频单独保存。不会复制整个数据集进个人目录。

- `Q0_anchor`：现有 15 任务 × L0–L3 × 50 = 3000 条仿真轨迹，以及 15 × 50 = 750 条人类参考。原始动作逐 episode 检查有限值/长度；视频按共享视频文件抽样检查，不宣称重新逐帧验收全部基准。
- `Q1_action`：外部动作或运动轨迹通过可观测的数值/时间/视频检查。
- `Q2_video`：仅用于视频辅助学习，不作为动作监督。
- `rejected`：带原因的排除清单。质量等级不同于仿真的 L0–L3。

EgoDex：只读取训练 part1/part2。手腕置信度 >= .9 的维度才监督；捏合距离还要求指尖置信度 >= .8。至少 80% 双腕时刻有效，手腕速度 p99 < 3 m/s。低置信度维度有 mask；缺少置信度则降为视频数据。20 维标签是未来观测到的双手腕 xyz/6D 旋转/捏合距离，整个预测 chunk 均表达在当前时刻的相机坐标系。它们不是人工测量的无误差真值，也不是机器人控制指令。

AgiBot：保留双 7DoF 手臂 + 两个夹爪的接口，拒绝移动底盘/明显腰部运动、时间异常、过大关节跳变。有效控制来源时刻来自 `action/*/index`。只用头部 RGB；当前不使用畸变更大的鱼眼流。刷洗水瓶仅截取标注 Brush 阶段，不包含倒水步骤。

本地 Alpha 数据存在关键限制：已检查的 `action/joint/position` 与同帧 `state/joint/position` 完全相同，不能用它们的零差值证明跟踪精度。因此本路线使用 `agibot_dual_gripper_future16`：给定 t 时刻观测，预测 t+1 起的未来测量关节轨迹和发布的夹爪信号，作为运动辅助监督，损失权重 .5。它不是已经验证可直接部署的独立电机控制标签。最终 manifest 逐条记录 `joint_action_equals_state` 与 `label_kind`；动作数组保留源字段，数据适配器明确执行未来时间偏移。原仿真的控制监督不作此偏移。

EPIC-KITCHENS：只从官方 train 标注和已完成下载的视频中选择有明确操作时间段的相关片段；排除重叠片段，限制每视频数量。没有连续机器人动作标签，因此仅用于视频辅助学习。

EPIC 候选池还按对象白名单复筛：切菜/水果、清洗餐具或食物、擦拭表面、折布巾。排除折滤纸/包装袋、切包装、擦面包、洗手等片段。人工查看的样例单独记录，不能外推成全部片段都已人工验收。

外部所有入选视频检查容器长度及开头/中间/结尾三帧。画廊样例额外完整解码。没有运动捕捉真值可计算全数据绝对姿态误差，Q1 表示自动质量筛选，不是“误差已证实小于某厘米”。

## 数据隔离与参考关系

原始训练 checkpoint 的配置已核对为 `human_train_config_15.json`、`sim_train_config_15.json`。保留原 seen/unseen 任务评测列表，并增加已有的 seen-heldout 人类示范口径。

受保护的 unseen：H4 遥控器收纳、H13 扫码、H20 倒水、H21 清理食物垃圾、H30 纸箱折叠/封箱。H30 的人类文本与仿真文本存在描述差异，因此保守排除两类外部内容。文件名 `PickFood` 不能简单理解为任意食物抓取。官方 test 数据不训练；外部同目标数据按任务类别和文本保守排除，但自动文本审查不能证明每帧不存在未标注的相关子动作。

EgoDex 以原录制 session 分组切验证集；EPIC 以参与者分组；AgiBot 以任务内连续收集批次分组。AgiBot 的批次是场景变化的代理，不是已核验的房间/操作者 ID。原有仿真训练数据全部保留，仿真成功率由 rollout 测量。所有归一化统计只使用训练数据，不读取 unseen 动作统计。

小批量 AgiBot 每任务至少保留两个训练批次以支持独立参考；拥有至少三个批次时，整组留出最接近该任务 15% 数量的一组。仅一至两个批次的任务本轮不另拆验证；不能把邻近轨迹切开后声称场景隔离。外部验证用于检查训练损失，最终主指标仍是原 seen/unseen 仿真成功率。

`reference_pairs.json` 列出真实的同目标候选对。不同 episode 不一定代表物理场景不同；其中 `different_scene` 来自可用场景元数据，明确区分两者。不把视频切出多个窗口再声称多了同样数量的独立示范。

## 建议的首轮实验

1. 数据检查、10–50 条轨迹过拟合与参考替换诊断。
2. 用同一个新架构跑仅原数据的基线，保留所有目标和评测环境。
3. 加外部机器人动作，再加人类动作，最后加无动作视频。分别记录增益，不把架构变化与数据增益混成一个结论。

最终默认采样比例：仿真动作 60%、原人类视频辅助 10%、AgiBot 15%、EgoDex 手部运动 10%、EgoDex 视频 2.5%、EPIC 视频 2.5%。每来源内部再按目标、episode 均衡，不让长轨迹或大任务支配采样。比例是采样占比，不是文件数量占比；缺失来源时只在已有来源重归一化。

可用 `--tiers Q0_anchor,Q1_action` 只训练原数据和通过动作检查的数据，或保留默认的全部等级。各等级文件夹保存 episode 索引，原视频和动作数组不重复复制。

首批限制每外部类别数量的原因：保留原基准作为训练中心，先确认高相关、不同视角/本体的数据是否产生正迁移。增加外部数据可能帮助折叠、清洗、放置等技能表征泛化，但不能预先保证 unseen 成功率提高。需要验证实际参考依赖，而不是模型靠当前场景猜任务。

报告 seen/unseen 各任务各 L 等级 success_once/success_at_end，再做任务宏平均；主实验至少 3 个随机种子，并给置信区间。公平对比同时记录总更新数、原仿真样本曝光量和训练耗时。总步数相同但原仿真占比下降的实验，不能单独用于断言数据没收益。

## 批量扩展筛选

扩量使用 `batch_curate.py`。第一批 `curate.py` 保留作筛选过程追溯；不再用反复逐片段解码的方式扩量。流程为：元数据任务/对象/未完成下载过滤 → 每批 128 条数值检查 → 合格候选的视频三帧检查 → 仅为入选记录保存动作数组。默认 4 个并行工作线程，按任务设配额；之后按任务、质量等级和异常类型抽查，不逐视频人工验收。

SQLite 缓存通过和失败的检查，源文件大小/修改时间、ZIP CRC、规则版本参与缓存键。修改数值筛选规则时更新 `RULE_VERSION`，修改视频判据时更新视频规则版本。`--retry-rejects` 可单独重试失败项。每任务结束保存断点。TAR 的 MP4 偏移索引会持久化并记录扫描位置，后续可继续扫描并直接读取视频成员，避免反复遍历深度图；初次索引新大归档仍有 I/O 成本。

```bash
# 生成下一批候选；与基准合并、执行 finalize 和抽查后才能作为正式训练清单。
.venv/bin/python -m examples.baselines.hypericlzq.batch_curate \
  --output /mnt/sdf/datasets/hypericlzq_candidates_v2 \
  --cache /mnt/sdf/datasets/hypericlzq_screen_cache_v1 \
  --seed-manifest /mnt/sdf/datasets/hypericlzq_curated_v1/manifest.json \
  --batch-size 128 --workers 4 --per-task 100 --candidate-limit 800
```

已经进行三来源的小批测试：47 条数值候选、44 个视频，首次约 14.90 秒；复跑约 1.15 秒，两阶段全部缓存命中、0 次重新解码。测试使用每批 8 条/4 线程，已复用现有 AgiBot 偏移；这些耗时不能外推到首次扫描全部 TAR。测试输出在 `/mnt/sdf/datasets/hypericlzq_batch_check_v1/`，没有并入正式的 4,919 条清单。16 个测试已通过，包含并行执行、缓存失效、失败缓存及 PAX/GNU TAR 断点索引。

## 使用

在 PolicyGen 根目录：

```bash
# 可选但推荐：缓存每个训练/辅助视频的逐帧冻结特征，避免每步重复解码和跑 DINO。
.venv/bin/python -m examples.baselines.hypericlzq.cache_features \
  --manifest /mnt/sdf/datasets/hypericlzq_curated_v1/manifest.json \
  --output /mnt/sdf/datasets/hypericlzq_curated_v1/dino_cache

# 初步训练。去掉 --cached-tokens 并使用原 manifest 可在线提取特征。
.venv/bin/python -m examples.baselines.hypericlzq.train \
  --manifest /mnt/sdf/datasets/hypericlzq_curated_v1/dino_cache/manifest.json \
  --output runs/hypericlzq_pilot --cached-tokens --batch-size 8 --accumulate 2 --steps 20000

# 仅原数据的新架构基线：其他超参数保持一致。
# 加 --source-ratios imitator_sim=.9,imitator_human=.1

# 多卡入口
.venv/bin/torchrun --nproc-per-node=4 -m examples.baselines.hypericlzq.train \
  --manifest /mnt/sdf/datasets/hypericlzq_curated_v1/dino_cache/manifest.json \
  --output runs/hypericlzq_ddp --cached-tokens --batch-size 8

# 相同任务和 L0–L3 的仿真指标
.venv/bin/python -m examples.baselines.hypericlzq.evaluate \
  --checkpoint runs/hypericlzq_pilot/last.pt --suite unseen --output results/hypericlzq_unseen
# --suite seen / seen_heldout 分别运行，禁止将三者混成一个成功率。
# 参考依赖诊断：分别加 --reference-mode wrong_task / blank。
# 正确参考若没有明显优于错误参考，需先解决模型忽略参考的问题。

.venv/bin/python -m unittest discover -s examples/baselines/hypericlzq/tests -v
```

`--resume` 恢复同配置模型/优化器/调度器；不承诺逐样本精确恢复。`--init-from` 仅加载本路线权重，允许修改损失权重并重新初始化优化器，可用于辅助预训练后联合训练。模型结构和动作空间/归一化必须一致。旧 hypericl checkpoint 与本路线结构不同，不能直接 resume。

训练包含 BF16、梯度累积、warmup+cosine、验证和原子 checkpoint 保存。原始 simulator 成功率需要真实 rollout，离线 validation loss 不能替代。当前验证日志是固定种子的局部样本 loss；DDP 日志标量来自 rank 0，动作梯度按全局有标签样本数校正。

官方来源：[EgoDex 数据格式](https://github.com/apple-aiml-research/ml-egodex)、[AgiBot Alpha 动作字段](https://huggingface.co/datasets/agibot-world/AgiBotWorld-Alpha/blob/main/README.md)。选择数量、阈值和等级是本项目的筛选决定，不是数据发布者对物理精度的保证。
