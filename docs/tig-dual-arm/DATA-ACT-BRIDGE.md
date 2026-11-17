# 同步轨迹到单任务 ACT：首轮最小闭环

2026-11-17。本轮把已验收的同步专家轨迹接到实际 LeRobot 数据、ACT 训练、严格重载与模型在线执行，对应实施方案 A／H／I。范围固定为药瓶分区 `TwoRobotPickPillToRegions-v1`、双 Panda、MPLib seed401 的一条171步轨迹。

## 实际结果

| 阶段 | 本轮结果 |
| --- | --- |
| 来源 | 原生生成和严格动作回放通过；171条原动作、172帧状态；原报告和评分不变 |
| 观测补录 | 172帧真实策略相机 `zed2i` RGB224×224、172条18维实际关节位置；恢复状态后渲染，物理推进0步 |
| 数据转换 | LeRobot image／Parquet，20 FPS；171条有效训练记录，全episode0训练；终帧单独保存 |
| 数据检查 | 实际reader／DataLoader通过；全部171条RGB、qpos和动作与导出一致；172帧qpos与原状态一致 |
| ACT训练 | 默认ResNet18及ACT主体，obs1／pred16／batch8／1epoch；22次更新使用全部171样本，182个参数张量发生变化，损失和梯度有限 |
| 严格重载 | `strict=True`；固定输入0／85／170的预测完全一致，最大绝对误差0 |
| 模型在线执行 | 200条模型动作、201帧图像／qpos／场景状态；相机参数及初始图像／qpos与离线完全一致 |
| 官方任务结果 | 曾经成功=false，最终成功=false，超时=true；接口链路完成=true |

训练及模型执行的代码提交为 `a0f8503`；导出、转换在提交前运行，其文件SHA与该提交中的对应源码一致。物理专家来源的运行提交为 `f390fdf`。各阶段原始回执、输入和源码SHA保留在本地证据目录；[可移植摘要](evidence/data-act-bridge-20261117.json)记录结果与原件身份。

这是一条轨迹的接口实验。策略在本次200步内没有完成任务，不将该模型运行计为合格专家数据。原专家的物理成功引用原生及动作回放证据；逐状态渲染只提供派生观测。尚未据此比较数据收益或泛化性能。

## 数据和控制契约

- 状态：实际左Panda的9维qpos，再接右Panda的9维qpos。每臂7个旋转关节和两根物理手指；归档31维articulation中的切片为 `[13:22]`。
- 动作：原16维指令不变，每臂7个关节目标和1个夹爪命令，`pd_joint_pos`、20Hz。状态中的两指位置不作为两个动作标签。
- 第 `t` 条训练记录是执行前的 `(RGB_t, qpos_t, action_t)`，`t=0..170`。终态索引171没有动作标签，独立保存在 `meta/terminal-observation.npz`。
- 图像使用原生策略相机224×224 RGB，uint8 HWC转float32 CHW并除255；训练和在线共用adapter，不做增强或额外ImageNet标准化。
- 16步动作块的偏移为 `i/20` 秒。越过episode末尾的位置由真实LeRobot reader标为padding，并同时进入L1损失及CVAE encoder attention mask。
- `bounds_q99`归一化只从171条训练记录计算，checkpoint携带相同统计供在线使用。单episode全部训练，不执行95/5随机划分。
- 在线每周期预测16步，执行第一步，再读取真实观测；只恢复一次源初态，无专家动作、规划器兜底或后续状态注入。连续夹爪预测保留，只按环境Box范围裁剪，裁剪前后值均记录。

## 代码改动

新增四个模块：

- `mani_skill/examples/motionplanning/dual/export_observations.py`：源认证、逐状态补录、相机与来源信息。
- `examples/baselines/lerobot_dataset/dual_arm_to_lerobot.py`：LeRobot转换、终帧、划分和行级来源。
- `examples/baselines/act/dual_arm_adapter.py`：离线／在线共用的RGB、qpos、归一化和动作契约。
- `examples/baselines/act/dual_arm_downstream.py`：数据检查、真实训练、严格重载及在线执行四个子命令。

同时修复原 `train_act_single.py` 的三处接线错误：导入能够接收dummy video的已有DETRVAE、使用已声明的 `obs_proj`、使条件输入维度与hidden维度一致。已有 `detr_video/detr_vae.py`增加可选padding mask，默认无mask调用保持兼容。未替换为其他策略，也未缩小默认模型。

## 运行方式

从此checkout根目录运行，使用已配置的TIG环境和资产。示例中的解释器、路径和GPU UUID需替换为实际值；输出目录应尚不存在。

```bash
export TIG_WORKSPACE=/path/to/dependency-workspace
export PYTHONPATH="$PWD"
export MS_ASSET_DIR="$TIG_WORKSPACE/infrastructure/maniskill"
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
export CUDA_VISIBLE_DEVICES='<available GPU UUID>'
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export TORCH_HOME=/path/to/torch-cache
export SOURCE_TRAJECTORY=/path/to/accepted-pill-seed401-recording
export BRIDGE_OUTPUT=/path/to/new-bridge-output
unset MANI_SKILL_L1 MANI_SKILL_L2 MANI_SKILL_L3
```

ACT使用torchvision的ResNet18 `IMAGENET1K_V1`权重。先在相同 `TORCH_HOME`准备其标准缓存；训练入口不静默替换成随机视觉骨干：

```bash
python -c 'from torchvision.models import resnet18, ResNet18_Weights; resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)'

python -m mani_skill.examples.motionplanning.dual.export_observations \
  --source-dir "$SOURCE_TRAJECTORY" --output-dir "$BRIDGE_OUTPUT/observations"

python -m examples.baselines.lerobot_dataset.dual_arm_to_lerobot \
  --input-dir "$BRIDGE_OUTPUT/observations" --output-dir "$BRIDGE_OUTPUT/lerobot" \
  --repo-id local/tig-pill-sync-seed401

python -m examples.baselines.act.dual_arm_downstream check \
  --export-dir "$BRIDGE_OUTPUT/observations" --dataset-dir "$BRIDGE_OUTPUT/lerobot" \
  --output-dir "$BRIDGE_OUTPUT/dataset-check" --device cpu

python -m examples.baselines.act.dual_arm_downstream train \
  --export-dir "$BRIDGE_OUTPUT/observations" --dataset-dir "$BRIDGE_OUTPUT/lerobot" \
  --output-dir "$BRIDGE_OUTPUT/train" --device cuda

python -m examples.baselines.act.dual_arm_downstream reload \
  --export-dir "$BRIDGE_OUTPUT/observations" --dataset-dir "$BRIDGE_OUTPUT/lerobot" \
  --checkpoint "$BRIDGE_OUTPUT/train/checkpoint_epoch_0001.pt" \
  --output-dir "$BRIDGE_OUTPUT/reload" --device cuda

python -m examples.baselines.act.dual_arm_downstream evaluate \
  --export-dir "$BRIDGE_OUTPUT/observations" \
  --checkpoint "$BRIDGE_OUTPUT/train/checkpoint_epoch_0001.pt" \
  --output-dir "$BRIDGE_OUTPUT/rollout" --device cuda
```

三个新增入口均提供 `--self-test`；它们的CPU测试覆盖时序、真实LeRobot写读与batch，以及完整默认ACT前向、padding不变性和无mask兼容。图像与数据写入源目录、覆盖已有导出／checkpoint／rollout均被拒绝。

输出中 `manifest.json`保存观测配置和来源；`meta/dataset-selection.json`为单episode转换划分配置；训练参数和归一化保存在checkpoint及 `training-report.json`；在线配置、任务结果和实际动作记录分别在 `rollout-report.json`及 `rollout.h5`。大型数据、权重、模型和本地运行环境不纳入Git。

## 后续

下一批先固定少量episode与独立评测初态，复用同一条导出、训练和执行链路，再扩到其他任务。需要判断同步数据的学习收益时，固定模型、数据量和训练预算做配对比较。人类视频条件ACT另核对示范来源、任务映射和配对层级。
