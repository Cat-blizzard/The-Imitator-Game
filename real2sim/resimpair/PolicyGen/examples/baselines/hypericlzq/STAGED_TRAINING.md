# 缓存与分阶段训练（2026-09-26）

入口为 `pipeline.py`，训练实现为 `train_staged.py`。旧 `train.py` 和旧 checkpoint 仍可使用；新实验从头训练基础策略，不把原联合训练的 nominal epoch 当作基础预训练轮数。

## 当前运行计划

**2026-09-26 稳定性排查更新：原 `staged_eef_v32_20260926` 正式任务已因 hyper 失稳停止，不能直接恢复其失败 `hyper/last.pt`。** 全量特征缓存已完成。以下20/20/20轮是目标预算，不表示训练健康完成。健康的隔离验证在原运行目录的 `stability_debug/guarded_hyper_probe/`，由同一个健康 base 第6轮权重开始。根据用户的新要求，进展统一按 epoch 记录，验证将衔接独立 A/B 专家监督对照；详见 [EXPERT_COMPARISON.md](EXPERT_COMPARISON.md)。原失败模型始终保留，不能误续训。验证结果尚不能代替闭环成功率。

此次修复包括：LoRA增量矩阵运算真正关闭autocast、动作DiT注意力显式FP32计算、生成参数增量软约束，以及optimizer更新前的有限大梯度/动作loss/增量范数异常检查。固定训练探针严重退化也会停止。详细证据和限制见运行目录的 `stability_debug/README.md`。

公共输出：`/mnt/sdf/datasets/hypericlzq_runs/staged_eef_v32_20260926`

全帧缓存：`/mnt/sdf/datasets/hypericlzq_feature_cache/dinov2_l_bf16_v2`

失稳前采用运行目录的 `run_pipeline.sh`：先完成全量缓存，再由 `--adaptive-resources` 调度训练。`optimization/training_ready.json` 仅记录当时的短程实现验证，**不代表长期稳定性**。失稳版本源码固定在 `source_snapshot_fast_v1` 并保留，不能用它启动修复版。

| 阶段 | epoch | micro batch | 累积 | 有效 batch | 数据与可训练部分 |
|---|---:|---:|---:|---:|---|
| base | 20 | 512 | 1 | 512 | 原始 15 任务；动作策略、动作适配器、观测/状态编码器 |
| hyper | 20 | 64/128/256 | 4/2/1 | 256 | 原始 15 任务；冻结基础策略及观测编码器，训练视频编码器、编译器/decoder、参数字典 |
| joint | 20 | 64/128/256 | 4/2/1 | 256 | 基础策略小学习率微调，逐步引入其他数据、未来特征和顺序辅助任务 |

base/hyper 的一轮为原始 3,000 条机器人轨迹全部 1,033,711 个查询帧（含末尾被 mask 的动作窗口），shuffle 无放回。每轮分别 2,019 和 4,038 次更新。joint 的一轮以期望抽取 1,033,711 个原任务查询计量，并记录实际的 `anchor_frame_equivalent_epoch`；它不是全混合数据的无放回遍历。

每 5 轮保存不可覆盖的 `stage/epoch_0005.pt` 等文件；每轮及每 500 更新保存可恢复 `last.pt`。SIGINT/SIGTERM 会在当前更新完成后保存再退出。分阶段目录分别为 `base/`、`hyper/`、`joint/`，各自的 epoch 从 1 计数。

2026-09-26 用户将 joint 从 10 轮延长到 20 轮，总更新数为 113,377（比原计划增加 57,690）。joint 尚未启动，学习率 warmup/cosine 将从启动时按完整 20 轮预算计算；保留第 5、10、15、20 轮 checkpoint。

优化前的耗时估计：base 20 轮实测约 3.23 小时，每轮 9.1–10.3 分钟；GPU1 共享条件下 hyper 20 轮预计 24–32 小时，joint 20 轮预计 42–60 小时。这些不是优化后的测量。最新真实数据短程测速、显存及训练恢复验证保存在运行目录 `optimization/`，完整生产 epoch 才能校准最终 ETA。

该次 base 在第 7 轮后出现训练退化：mean loss 第 7 轮为 0.0566，第 20 轮为 1.7798；原任务训练探针运动位置误差第 6 轮约 23.8mm，第 20 轮约 80.1mm。CPU FP32 复核也能复现性能下降，后期 DiT 条件调制/attention 输入幅度明显增长，不能将问题归为单纯日志或 BF16 显示误差。退化的完整因果仍未确定。当前 `--init-from-best` 使用附加 Panda 验证集最优的第 6 轮 `base/best_panda.pt` 初始化 hyper；后续阶段也从前阶段验证最佳 checkpoint 初始化。所有最终轮/每5轮 checkpoint 仍保留，离线误差不等于闭环成功率。

## 当前提速措施

- 缓存：GPU0、图像 batch256，有界解码预取；只传 uint8 图像到 GPU；有界后台刷盘，数据完成写入后才发布元信息，所有写入完成后才发布完整 manifest。沿用 v2 特征格式及逐帧时间戳规则，复用已完成文件。
- 参考复用：原平台同任务在一个 microbatch 共享两段示范，每个 batch 重新抽样；遇到查询同记录或弱跨任务配对时回退独立抽样。外部数据的 self-reference/跨来源规则保留。E 按参考去重，T/R/D 按参考与目标动作空间去重，观测驱动的阶段选择仍逐查询计算。域专属 LayerNorm 的 affine 权重按样本选择，人类/机器人输入可在一次 E 前向中处理；与分域执行有输出和梯度对照。任务 ID 仅用于采样/分组和损失标签，不作为模型输入。
- 动作网络编译：可用 `--compile-target` 编译 DiT 的 forward，使用动态 batch 以支持混合数据中的无动作样本和末尾小批量；checkpoint 参数名称不变。只在实际训练/评估/恢复验证通过后启用。BF16 编译与 eager 不宣称逐位相同，另有 loss/梯度数值对照。
- 资源调度：每 5 epoch 保存后退出当前训练子进程，重新查看 GPU 空闲显存和利用率，选较空闲设备续训；显存探测结果再留安全余量。按 256×1、128×2、64×4 切换，保持有效 batch256、epoch 查询量和学习率进度；中途恢复会校验有效 batch。不会终止其他任务。
- 训练质量：使用已验证更好的基础权重；保留原任务采样比例、全部辅助损失及20/20/20轮预算。提速本身不证明成功率恢复，需要同一参考、种子、任务和参数保持32步协议的闭环评估。

基础阶段保留 EEF 相对位移表示与现有训练集归一化，未来预测是 DINO 特征而不是 RGB 像素。没有在本次实现中增加原版任务专家 bank/参数蒸馏，不能把这三个阶段称为原版 A–E 的完全复刻。

## 批量与缓存

最初 GPU1 与其他任务共享时，生产维度 BF16 测试包含反传及 AdamW 状态：

- DINO 的内部图像批量从 4 提高至 64。测试峰值 allocated 约 1.81GiB、reserved 2.16GiB。
- 基础策略 batch512 的真实缓存短程测试峰值约 1.03GiB。虽测试到 1024 仍可运行，正式选择 512，兼顾更新次数、读取速度与缓存进程并发。
- hyper batch128、累积2 的真实短程测试峰值约 3.26GiB。batch256 合成测试已接近 6GiB 预算上限，因此未选择。
- joint batch128、累积2 的真实短程测试峰值约 4.98GiB。

这些数值不是全数据吞吐承诺。磁盘访问、解码和其他 GPU 任务会改变实际速度；`metrics.jsonl` 另记录 `data_wait_seconds` 和 `seconds_per_update`。

完整原始机器人+人类示范缓存约 68.6GiB；预先还缓存附加任务的 held-out Panda 数据用于验证。全 manifest 的 FP16 缓存约 445.4GiB（不计少量文件头/元数据）。DINO 冻结，使用 BF16 前向、FP16 存储；可训练的 E/T/R/D 不做跨更新缓存。

顺序解码逐帧提取一次，包含完整时间轴。首个新来源的起/中/末帧会与随机时间戳读取对照。修复了 MP4 B-frame 情况下 exact seek 返回后一帧的问题：先从更早的关键帧解码，再按显示时间戳选择。文件按源区间及配置识别，有 shape/finite 检查，写入完成后原子发布，重启可复用已完成文件。

原调度中基础训练与扩展缓存并行，hyper/joint 上限为 5.9GiB；最新调度遵照用户要求先完成全量缓存，再根据实际资源分配训练，不再固定该上限。正式训练不再在线解码视频或运行 DINO。

## 计算复用

同一个 forward 中，reference/reference2 的 E 输出在动作、一致性和对比损失间复用，保留完整梯度。只需要任务表征的对比损失不再运行 T/R。基础阶段只读取当前观测特征，不读取无用示范、上下文和未来帧。

## 混合比例与验证

原失稳 hyper 目标为 `action + .02 task + .02 consistency`；没有专家参数蒸馏或函数蒸馏。修复版 staged hyper/joint 加入 `.01 parameter`：按层计算 `max(||ΔW||²/||W0||² - .25², 0)`，按参考视频取平均、按层求和，并约束AdaLN bias增量的均方幅度。这是对过大生成增量的软正则，不是任务LoRA教师监督，也不是把输出动作裁小。真实低秩乘积含交叉项，用Gram矩阵精确计算范数。旧checkpoint默认该权重为0；改变目标后必须在新目录 weights-only 初始化，不能直接恢复旧optimizer配置。

新增保护默认在裁剪前梯度范数超过1000、任一生成权重增量/基础权重范数超过2、动作loss相对近期基线发生灾难性跃升时，在 `optimizer.step()` 前停止并保存诊断checkpoint。固定训练探针action loss超过历史最好5倍且大于0.5时也停止。保护参数为 `--max-grad-norm` / `--max-delta-ratio`；实际梯度裁剪仍为1。异常checkpoint禁止直接resume，pipeline不会进入下一个阶段。阈值针对严重失稳，不是提前停止一般的验证波动。

joint 第 1 轮原任务 90%、额外仿真 10%；第 2 轮线性过渡，第 3 轮达到：原任务70%、额外仿真15%、OXE8%、AgiBot2%、原平台人类视频2%、EgoDex动作2.5%、EgoDex无动作0.25%、EPIC0.25%。辅助损失在前两轮更新中渐进启用；基础参数学习率为其他可训练参数的 0.1 倍。

训练集/验证集划分、动作表示和归一化未更换。基础与 hyper 阶段从训练引用池取参考，不使用验证视频训练。每轮记录两个独立探针：原任务训练帧的 fitting probe（不叫验证集），附加任务 held-out Panda 的验证探针。验证特征来自修复采样后的新缓存，不直接复用旧运行的图像特征。物理误差与闭环成功率分开。

评估器支持基础阶段（完全绕过 hypernet），hyper/joint 仍按完整参数保持 32 个控制步的协议评估；当前观测每步更新。

## 启动与恢复

在 PolicyGen 根目录：

```bash
.venv/bin/python -u -m examples.baselines.hypericlzq.pipeline --gpu auto --cache-gpu 0 \
  --adaptive-resources --share-references --init-from-best --compile-target \
  --snapshot-name source_snapshot_fast_v1 --tokenizer-batch 256 \
  --base-epochs 20 --hyper-epochs 20 --joint-epochs 20
```

同一 output 有进程锁防止重复启动。supervisor 保存源码快照并从快照运行子进程；阶段之间传递权重，恢复同一阶段时加载优化器、查询位置和 RNG。恢复精确保留剩余帧索引，但 DataLoader worker 的随机参考/辅助片段流会重启，不声称逐位等价恢复所有辅助任务。阶段数据或 batch/epoch 配置改变时，不能直接复用旧阶段的 optimizer resume。

查看 `pipeline_status.json`、`pipeline_plan.json`、`cache_anchors.log`、`cache_all.log`、各阶段的 `metrics.jsonl`/`epochs.jsonl`。错误会记录到 supervisor 状态，不会把失败的阶段当成完成继续传递。

原联合实验第 60 个 nominal epoch 已保留在 `eef_v31_20260925/epoch_0060.pt`。`smoke/` 是接口/内存/恢复检查，不用于声称收敛或作为正式训练初始化。
