# HyperICLZQ 第一批数据与架构报告

本轮完成新实现、数据筛选、可播放样例和链路验证；**尚未正式训练，没有 seen/unseen 成功率结论**。代码在 `PolicyGen/examples/baselines/hypericlzq/`，原 `hypericl/` 未被本次修改。

正式清单与精选样例目录：`/mnt/sdf/datasets/hypericlzq_curated_v1/`。原始视频仍引用既有公共数据目录/归档；没有重复拷贝数 TB 原视频。训练唯一入口是 `manifest.json`，`*_records.json` 是筛选过程的原候选清单，不能作为最终训练清单。

## 1. 最终数量

共 **4,919** 条独立 episode/标注片段，其中原数据 **3,750** 条，新增 **1,169** 条。总计 **16.31 小时**，1,761,551 个采样帧。未用重复采样窗口扩大“示范数”。

| 来源/用途 | 等级 | 训练 | 外部验证 | 合计 |
|---|---|---:|---:|---:|
| 原 15 任务 × L0–L3 仿真控制 | Q0_anchor | 3000 | 0 | 3000 |
| 原 15 任务人类参考 | Q0_anchor | 750 | 0 | 750 |
| EgoDex 双腕运动 | Q1_action | 564 | 51 | 615 |
| EgoDex 仅视频 | Q2_video | 173 | 23 | 196 |
| AgiBot 未来关节运动 | Q1_action | 210 | 31 | 241 |
| EPIC 操作片段 | Q2_video | 107 | 10 | 117 |
| **合计** | | **4804** | **115** | **4919** |

训练中 3774 条有控制或运动标签，其余提供参考与视频辅助任务。原仿真 3,000 条全部保留；外部验证不替代原 L0–L3 仿真成功率测试。外部训练实际 1054 条。

| 数据源 | 类别/任务 | 总条数（含验证） | 有运动标签 |
|---|---|---:|---:|
| egodex | add_remove_lid | 140 | 100 |
| egodex | basic_fold | 40 | 0 |
| egodex | clean_cups | 211 | 180 |
| egodex | clean_surface | 160 | 120 |
| egodex | clean_tableware | 125 | 120 |
| egodex | fold_stack_unstack_unfold_cloths | 135 | 95 |
| agibot | 书架插书 | 50 | 50 |
| agibot | 折短裤 | 50 | 50 |
| agibot | 拿开杯子后擦桌面 | 50 | 50 |
| agibot | 放入洗碗机 | 3 | 3 |
| agibot | 水瓶刷洗阶段 | 38 | 38 |
| agibot | 衣物收纳 | 50 | 50 |
| epic | cut | 9 | 0 |
| epic | fold | 1 | 0 |
| epic | stir | 46 | 0 |
| epic | wash | 55 | 0 |
| epic | wipe | 6 | 0 |

EgoDex 覆盖 207 个录制 session、16 种环境描述；EPIC 的 117 个不重叠片段来自 24 个完整本地视频、21 名参与者。环境描述数量不能等同于独立物理房间数量。

## 2. 架构为什么这样改

```text
参考视频 → 冻结 DINO → E 视频任务表征 → T/R 技能状态 → D 生成 LoRA
                                                       ↓
当前帧 → 冻结 DINO → 观测聚合 ──────────────→ 共享 Flow-Matching policy
当前本体状态、mask、目标动作空间、时间间隔 ───→              ↓
                                            当前本体专用动作输出接口
```

- 三套原生接口：仿真 Panda 16 维控制；EgoDex 20 维双腕运动；AgiBot 16 维未来关节运动/夹爪信号。用各自训练统计归一化，通过输入/输出适配层连接到共享 32 维动作 latent。坐标、单位、夹爪定义不混用。
- 当前观测和参考视频共享冻结 DINO。默认 E/T/R/D 可训练，policy 骨干和参数基也可训练；没有 task-ID 直接输入 policy。
- 默认启用动作 FM、未来特征、顺序、同目标对比、双参考行为一致性。未来预测只看目标之前的上下文，预测间隔为 0.2/0.5/1/2 秒；可加 32×32 RGB 预测。记录复制最后一帧特征的基线。
- 有动作/运动标签的数据可同时训练 hypernet 和 policy；无动作视频只训练任务表征及辅助头，不伪造零动作。动作 FM 的梯度能传到 D/R/T/E。
- 同目标双参考一致性是在同一观测、噪声和 FM 时间下匹配两套生成策略的速度场，不要求 LoRA 参数逐元素相等。当前路线没有专家参数蒸馏、世界模型想象或测试时更新。
- 默认总可训练参数 **34,031,557**；E/T/R/D 合计 **18,434,959**；共享 DiT/参数基 **14,164,000**。冻结 DINO **304,368,640** 参数不计入可训练量。

这里保留超网络的任务条件化能力，同时先用可检查的联合目标验证其作用。不能把额外模块数量当作成功率保证。

## 3. 参考视频怎样配对

原数据继续 human→sim；仿真 query 约 70% 选择对应人类参考，其余优先同目标不同 L 等级/episode。外部数据默认 20% 自参考，可直接“人类完整视频给 hypernet，当前人类帧和后续运动给 policy”；其余优先同目标不同录制组，存在场景元数据差异时优先选不同场景。

`reference_pairs.json` 给出 64 个目标组的真实候选示例，其中 15 组跨 human/robot 域、34 组场景描述不同。它们是目标组示例数，不是重复生成的训练视频数量。

具体外部配对：`ego_part1_clean_tableware_1491`（红色背景）作为参考，`ego_part1_clean_tableware_1051`（白色背景）提供当前帧和双腕运动；都是同类餐具的清洗，录制 session 不同。开盖/盖盖、折叠/展开方向分开分组。

**首批没有强行把 AgiBot 与 Panda 的宽泛同类动作当作完整同目标配对。** 例如“折短裤”和“折毛巾后放入篮子”的完整目标不同；“刷瓶子”和“刷烧杯三次”的操作也有差异。当前代码支持不同机器人参考→指定目标本体，但首批新增数据主要通过共享表征和运动辅助训练产生迁移。未来只有确认完整目标等价后才合并 `goal_key`。新本体还需要自己的动作 schema/适配层，不能保证零样本直接控制任意机械臂。

## 4. 选择方法与质量边界

Q0_anchor 是保留的原基准；Q1_action 是通过数值检查的控制/运动数据；Q2_video 不做动作监督；rejected 保存原因。它们与场景难度 L0–L3 不是同一套等级。

EgoDex 仅训练 part1/part2：有穷姿态、2–90 秒、手腕置信度 ≥0.9，至少 80% 双腕时刻有效，平移速度 p99 <3 m/s。捏合监督还要求指尖置信度 ≥0.8。整个未来动作 chunk 表达在当前相机坐标系，避免未来头部运动改写动作坐标。MP4 CRC+大小用于精确视频去重，每录制 session 限 12 条。缺置信度的数据降为视频任务。

EgoDex **不是机器人电机控制真值**。所选 Q1 的捏合维度有效覆盖率中位数仅 **2.0%**，主要可靠信息来自双腕而非精细抓握；低置信度维度被 mask，动作损失权重为 .35。低平移速度片段另外核查手腕旋转和图像，未用单一平移阈值误删清洗动作，见 `motion_review.json`。

AgiBot 从六任务各自首个本地归档中筛选，每任务最多 50 条；只保留双 7DoF + 两夹爪，使用头部 RGB。检查约 30Hz 单调时间、p99 间隔 ≤60ms、关节有效帧覆盖、关节单步跳变、低底盘速度和稳定腰部；不混入需要底盘控制或灵巧手的样本。刷瓶子只取 Brush 标注段，排除其倒水子过程。241 条全部存在 **joint action 与同帧 state 完全相同** 的本地字段现象。因此不能把零差值称为精确跟踪证据，使用 t+1 起的未来测量关节和夹爪信号作为辅助监督，权重 .5；不宣称可直接在 AgiBot 上部署。

所选 AgiBot 的关节单步绝对变化 p99 的轨迹间最大值为 **0.0224 rad/frame**。这是平滑性检查，不是末端位置精度。官方字段定义见 [AgiBot Alpha README](https://huggingface.co/datasets/agibot-world/AgiBotWorld-Alpha/blob/main/README.md)；本地字段相等的结论来自逐条数组检查。

EPIC 只读官方 train 标注和无 `.aria2` 的完整视频，片段 2–30 秒，不重叠，每原视频限 12 条；先按动作挑选，再按对象白名单和样例复核收紧。144 个候选最终留 117 个，剔除折滤纸/包装袋、切包装、洗手、擦面包及一个操作区域明显出画的样例。没有连续动作标签，仅参与辅助任务。

外部所有入选片段检查视频长度和首/中/尾三帧；24 个画廊样例额外完整解码。人工/视觉复核的具体样例记在 `visual_review.json`，不等于全数据逐帧人工验收。当前不能给出厘米级动作绝对误差，也不能证明每个演示最终完成了任务。

本轮排除原因计数：`{"invalid_pose_or_duration": 12, "irregular_timestamps": 82, "too_few_valid_control_frames": 2, "manual_review_contact_out_of_view": 1, "object_not_in_first_batch_scope": 26}`。数量上限和 session 上限跳过的候选不记作质量不合格，所以不能据此倒推全源数据合格率。

HoloAssist 未完整下载的文件、未完成语义配对的 HOT3D、粗粒度 Charades 不进入第一批；OXE 仍按此前要求不下载。此轮没有启动任何新数据下载。

## 5. unseen 隔离

保留原 15 任务训练和 seen/unseen L0–L3 测试配置。H4 遥控器收纳、H13 扫码、H20 倒水、H21 清理食物垃圾、H30 纸箱折叠/封箱受保护；外部同目标数据按类别与文本保守排除。H21 的 `PickFood` 文件名不代表所有食物操作，H30 人类/仿真文本不完全一致，已覆盖两种含义。

未使用官方 EgoDex test、EPIC val/test 或基准 unseen 动作统计。EgoDex 按录制 session，EPIC 按参与者划分；AgiBot 按连续 30 条收集块，至少保留两个训练组，三个以上组时整组留出接近 15% 的一组。收集块是场景隔离的代理，不是核验过的房间 ID。

原 seen 测试照旧保留，同时提供 seen_heldout 人类参考口径，避免把“见过训练参考”与“换新参考”混为一谈。最终报告各任务各 L 的 success_once/success_at_end，再求任务宏平均；推荐至少三随机种子和置信区间。自动文本过滤不能保证每帧不存在未标注的重叠子动作。

## 6. 为什么第一批是这个规模

新增 1169 条约占全部示范的 23.8%，其中训练新增 1054 条；原 3,750 条仍是主体。目标是先验证相关数据能否正迁移，不能预先从条数推算成功率增益。首个归档/每类别上限使这一批容易追溯、检查、定位负迁移；以后可扩展相同采样规则到更多 shard。

推荐采样：仿真 60%、原人类视频 10%、AgiBot 15%、EgoDex 双腕 10%、EgoDex 视频 2.5%、EPIC 2.5%；来源内部均衡目标和 episode。损失另加置信度权重，不按帧数让长视频支配训练。

可能改善 seen 的机制是：清洗、布料处理、容器操作、插入/收纳提供更多物体、材质和视角；原人类→仿真配对继续约束跨域任务对应。可能改善 unseen 的机制是：共享物体交互/时序表征、任务参考依赖和组合能力，且不喂 unseen 同目标数据。对于纸箱、扫码等差异很大的新任务，增益尤其不能保证。

建议按四个可比较实验推进：A=新架构仅原数据；B=A+外部机器人运动；C=B+人类运动；D=C+视频辅助。先用 10–50 条轨迹做过拟合与错误参考诊断，再跑基线和混合实验。动作阶段无须先逐任务训专家 LoRA。先采用 20k 更新作观测窗口，再根据成功率和验证趋势决定扩量。

对比同时控制总计算量和原仿真样本曝光量：原数据基线仿真采样 .9、混合 .6 时，匹配仿真曝光需混合多跑 1.5 倍步数（例如 30k 对 20k）；同总步数实验也要保留并记录耗时。若正确参考不优于错误参考、seen 持续退化或未来预测不优于复制基线，应先修复/减小辅助比例，而非继续盲目扩量。

## 7. 批量筛选与扩量

后续扩量使用代码目录的 `batch_curate.py`：先由任务类别、操作对象、unseen 隔离和下载完整性筛选元数据；默认每批 128 条、4 个并行工作线程做数值检查；只有通过数值检查的候选才进入视频三帧检查。入选后才写动作数组，人工仅按任务/质量层和异常抽查。

检查结果写入 SQLite（通过/失败均缓存）；源文件元信息和规则版本参与缓存键。TAR 视频成员索引记录字节偏移、PAX 状态和扫描断点，可复用第一批已知偏移。新归档首次仍需扫描，但后续不会从头重复扫描已完成部分。每个任务结束保存结果，避免晚期中断导致整批重做。

真实小批测试使用 4 线程、每批 8 条：47 条数值候选、44 个视频，首次约 14.90 秒；同批复跑约 1.15 秒，两阶段全命中缓存，0 次重新解码。测试复用了已有 AgiBot 偏移，不代表全量首次扫描也能获得同等倍数的提速。批处理测试结果未加入本次正式训练清单；详见 `batch_pipeline_check.json`。

## 8. 已验证与未验证

- 16 个单元测试通过：梯度链路、原生维度、置信度/填充 mask、当前相机坐标、AgiBot 未来偏移、参考/切分隔离、BF16、RGB、缓存与在线时间戳一致、权重加载，以及并行筛选缓存和 PAX/GNU TAR 断点索引。
- 所有 3856 个入选动作/运动数组完成 shape/有限值检查；真实数据接口抽查 20 个训练/验证样本，均通过。
- 四种真实来源 + 本地冻结 DINO 完成三步 CPU 反向传播及动作生成。检查时使用 112px/2 帧/8 patches 降低成本，核心网络维度保持默认；这是链路检查，不能当作正式训练收敛证据。
- 合成缓存数据的训练入口、梯度累积、验证、checkpoint 和 resume 已跑通。
- **尚未验证：正式多卡 DDP、完整特征缓存、正式训练和 ManiSkill rollout 成功率。** 旧 hypericl checkpoint 不能直接 resume 到新结构。

如果全部以默认 32×1024 FP16 缓存，预计特征数组约 **107.5 GiB**，尚未生成。缓存避免训练时反复解压/解码和跑冻结 DINO；首轮可先缓存需要的质量等级。运行命令见代码目录 `README.md`。

## 9. 视频样例

可直接打开 [全部样例画廊](/mnt/sdf/datasets/hypericlzq_curated_v1/gallery.html)。以下都是真实入选视频，非合成图像：

| 来源 | 说明 | 文件 |
|---|---|---|
| imitator_sim | human_H1 | [视频](/mnt/sdf/datasets/hypericlzq_curated_v1/samples/ig_sim_L0_TwoRobotStirSpoon-v1_0000.mp4) / [四帧预览](/mnt/sdf/datasets/hypericlzq_curated_v1/samples/ig_sim_L0_TwoRobotStirSpoon-v1_0000.jpg) |
| imitator_sim | human_H49 | [视频](/mnt/sdf/datasets/hypericlzq_curated_v1/samples/ig_sim_L0_TwoRobotFoldTowel-v1_0000.mp4) / [四帧预览](/mnt/sdf/datasets/hypericlzq_curated_v1/samples/ig_sim_L0_TwoRobotFoldTowel-v1_0000.jpg) |
| imitator_sim | human_H42 | [视频](/mnt/sdf/datasets/hypericlzq_curated_v1/samples/ig_sim_L0_TwoRobotCleanCup-v1_0000.mp4) / [四帧预览](/mnt/sdf/datasets/hypericlzq_curated_v1/samples/ig_sim_L0_TwoRobotCleanCup-v1_0000.jpg) |
| imitator_human | human_H1 | [视频](/mnt/sdf/datasets/hypericlzq_curated_v1/samples/ig_human_human_H1_0000.mp4) / [四帧预览](/mnt/sdf/datasets/hypericlzq_curated_v1/samples/ig_human_human_H1_0000.jpg) |
| imitator_human | human_H49 | [视频](/mnt/sdf/datasets/hypericlzq_curated_v1/samples/ig_human_human_H49_0000.mp4) / [四帧预览](/mnt/sdf/datasets/hypericlzq_curated_v1/samples/ig_human_human_H49_0000.jpg) |
| imitator_human | human_H42 | [视频](/mnt/sdf/datasets/hypericlzq_curated_v1/samples/ig_human_human_H42_0000.mp4) / [四帧预览](/mnt/sdf/datasets/hypericlzq_curated_v1/samples/ig_human_human_H42_0000.jpg) |
| egodex | Clean cups using a sponge and soapbottle while sitting at a pink tablecloth with a lavender background. | [视频](/mnt/sdf/datasets/hypericlzq_curated_v1/samples/ego_part1_clean_cups_1218.mp4) / [四帧预览](/mnt/sdf/datasets/hypericlzq_curated_v1/samples/ego_part1_clean_cups_1218.jpg) |
| egodex | Clean plates, forks, spoons, and knives using sponge and dish soap while sitting at a wooden table with a white background. | [视频](/mnt/sdf/datasets/hypericlzq_curated_v1/samples/ego_part1_clean_tableware_1051.mp4) / [四帧预览](/mnt/sdf/datasets/hypericlzq_curated_v1/samples/ego_part1_clean_tableware_1051.jpg) |
| egodex | Scrub the surface while sitting with no tablecloth. | [视频](/mnt/sdf/datasets/hypericlzq_curated_v1/samples/ego_part1_clean_surface_2325.mp4) / [四帧预览](/mnt/sdf/datasets/hypericlzq_curated_v1/samples/ego_part1_clean_surface_2325.jpg) |
| egodex | Place the lid onto the cup on a wooden table while sitting with a brown background. | [视频](/mnt/sdf/datasets/hypericlzq_curated_v1/samples/ego_part1_add_remove_lid_1512.mp4) / [四帧预览](/mnt/sdf/datasets/hypericlzq_curated_v1/samples/ego_part1_add_remove_lid_1512.jpg) |
| egodex | Fold a small tshirt on a red tablecloth while sitting. | [视频](/mnt/sdf/datasets/hypericlzq_curated_v1/samples/ego_part2_basic_fold_1360.mp4) / [四帧预览](/mnt/sdf/datasets/hypericlzq_curated_v1/samples/ego_part2_basic_fold_1360.jpg) |
| egodex | Unstack and unfold cloths on a green tablecloth while sitting against a brown background. | [视频](/mnt/sdf/datasets/hypericlzq_curated_v1/samples/ego_part2_fold_stack_unstack_unfold_cloths_905.mp4) / [四帧预览](/mnt/sdf/datasets/hypericlzq_curated_v1/samples/ego_part2_fold_stack_unstack_unfold_cloths_905.jpg) |
| agibot | Fold shorts | [视频](/mnt/sdf/datasets/hypericlzq_curated_v1/samples/agi_362_649567.mp4) / [四帧预览](/mnt/sdf/datasets/hypericlzq_curated_v1/samples/agi_362_649567.jpg) |
| agibot | Sort clothes | [视频](/mnt/sdf/datasets/hypericlzq_curated_v1/samples/agi_398_680010.mp4) / [四帧预览](/mnt/sdf/datasets/hypericlzq_curated_v1/samples/agi_398_680010.jpg) |
| agibot | Insert a book into the bookshelf | [视频](/mnt/sdf/datasets/hypericlzq_curated_v1/samples/agi_384_655324.mp4) / [四帧预览](/mnt/sdf/datasets/hypericlzq_curated_v1/samples/agi_384_655324.jpg) |
| agibot | Wash dishes with dishwasher | [视频](/mnt/sdf/datasets/hypericlzq_curated_v1/samples/agi_357_652951.mp4) / [四帧预览](/mnt/sdf/datasets/hypericlzq_curated_v1/samples/agi_357_652951.jpg) |
| agibot | Pick up the item to wipe away the stain. | [视频](/mnt/sdf/datasets/hypericlzq_curated_v1/samples/agi_421_678610.mp4) / [四帧预览](/mnt/sdf/datasets/hypericlzq_curated_v1/samples/agi_421_678610.jpg) |
| agibot | Brush inside the water bottle with a cup brush. | [视频](/mnt/sdf/datasets/hypericlzq_curated_v1/samples/agi_392_648676.mp4) / [四帧预览](/mnt/sdf/datasets/hypericlzq_curated_v1/samples/agi_392_648676.jpg) |
| epic | stir food | [视频](/mnt/sdf/datasets/hypericlzq_curated_v1/samples/epic_P01_18_625.mp4) / [四帧预览](/mnt/sdf/datasets/hypericlzq_curated_v1/samples/epic_P01_18_625.jpg) |
| epic | cut squash | [视频](/mnt/sdf/datasets/hypericlzq_curated_v1/samples/epic_P01_18_49.mp4) / [四帧预览](/mnt/sdf/datasets/hypericlzq_curated_v1/samples/epic_P01_18_49.jpg) |
| epic | wash plate | [视频](/mnt/sdf/datasets/hypericlzq_curated_v1/samples/epic_P12_04_80.mp4) / [四帧预览](/mnt/sdf/datasets/hypericlzq_curated_v1/samples/epic_P12_04_80.jpg) |
| epic | wipe hob | [视频](/mnt/sdf/datasets/hypericlzq_curated_v1/samples/epic_P05_08_132.mp4) / [四帧预览](/mnt/sdf/datasets/hypericlzq_curated_v1/samples/epic_P05_08_132.jpg) |
| epic | fold tea towel | [视频](/mnt/sdf/datasets/hypericlzq_curated_v1/samples/epic_P22_10_291.mp4) / [四帧预览](/mnt/sdf/datasets/hypericlzq_curated_v1/samples/epic_P22_10_291.jpg) |
| egodex | Clean plates, forks, spoons, and knives using sponge and dish soap on a wooden table with a red background while sitting. | [视频](/mnt/sdf/datasets/hypericlzq_curated_v1/samples/ego_part1_clean_tableware_1491.mp4) / [四帧预览](/mnt/sdf/datasets/hypericlzq_curated_v1/samples/ego_part1_clean_tableware_1491.jpg) |
