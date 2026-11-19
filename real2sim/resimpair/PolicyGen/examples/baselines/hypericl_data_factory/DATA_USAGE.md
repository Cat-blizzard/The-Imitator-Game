# 数据用途与动作约定

[统一数据和视频索引](/mnt/sdf/datasets/hypericl_data_factory_20261016_v1/DATASET_INDEX.md) · [最终详细报告](/mnt/sdf/datasets/hypericl_data_factory_20261016_v1/data_preparation_multitask_report_v10_with_start_state_aux/REPORT.md)

根目录：/mnt/sdf/datasets/hypericl_data_factory_20261016_v1/。当前最终清单为policy_multitask_append_v8_with_robot_start_state/policy_manifest.json，SHA256为5a7c5fe857e397acc6a7745679a9fbf9e21f18f3a17fc04f796694a62f9aee9c。

7409行包括原4050行、3337条新增机器人和22个人类窗口。3508正式尝试中的171失败独立保留，不进入成功action监督；起始姿态扩展的11次资格尝试排除。原有行和归一化保持，旧中间版本不替代最终读取证明。

## 四类用途与读取器

|用途|参考|执行与监督|约束|
|---|---|---|---|
|人类参考→机器人|原示范或批准的新窗口|机器人RGB、状态、EEF动作|同目标及必要阶段核验，保持split与typed准入|
|机器人参考→机器人|独立录制和布局的机器人视频|另一段机器人RGB、状态、EEF动作|同goal、不同group、同split、参考资格有效|
|同本体跨场景|同机器人本体的不同布局参考|机器人动作|与前一类重叠，不增加独立示范数|
|独立人类辅助|原生人类视频及单独前缀特征|腕部运动、未来视觉和各自mask|独立读取，不要求配套机器人动作|

前三种是策略route，独立辅助不是第四个route。

```python
from pathlib import Path
from examples.baselines.hypericl_data_factory.append_task_snapshots_typed_v2 import MultiTaskPolicyRouteDataset
from examples.baselines.hypericlzq.model import ZQConfig

manifest = Path('/mnt/sdf/datasets/hypericl_data_factory_20261016_v1/policy_multitask_append_v8_with_robot_start_state/policy_manifest.json')
data = MultiTaskPolicyRouteDataset(manifest, ZQConfig(frames=32),
                                  'human_to_robot', split='train')
sample = data[(0, data.rows[0]['length'] // 2)]
```

另两个route名为robot_to_robot、same_embodiment_cross_scene。没有合法参考时明确不可用，不跨split借用参考。goal/程序/场景ID用于选样和核验，不作为额外策略观测；特权物体状态与分割用于专家规划和QA。

参考输入32张224×224 RGB，执行输入单张224×224 RGB加机器人状态；原生512视频和时间轴保留。32输入帧与64运动监督时间点分别配置。

## 机器人动作和时间

Panda左右臂各7维，共14个有效分量：`[dx,dy,dz,rx,ry,rz,openness]`。平移单位米，旋转为SO(3)旋转向量、单位弧度，夹爪0闭合、1张开。各臂使用query时刻的panda_hand_tcp：

`dp = R_query.T @ (p_target - p_query)`

`dr = Log(R_query.T @ R_target)`

整个chunk相对同一query，不是相邻帧增量或速度。目标来自原生command[t]的实际URDF FK，target_start_offset=0；state/video[t]在执行command[t]之前记录，不能自行错移一帧。

保留原生16维双臂命令、18维实测关节状态、测量TCP、时间戳和相机信息。多动作空间tensor补齐到20维时，Panda额外6维以action_mask屏蔽。冻结归一化没有重拟合。

原基准30Hz和新增20Hz保留各自时间戳及action_dt，现有模型使用log1p(action_dt)，不盲目重采样；这不证明已经学会跨频率泛化。

## 人类运动和因果未来视觉

每只手分别由可靠原生世界系腕部位姿计算：

`T_relative = inv(T_world_from_wrist[query]) @ T_world_from_wrist[target]`

保存左右手、帧号、时间和有效性掩码。密集目标64×2×6，是腕部运动监督，不是Panda action。相机运动不直接混入世界系腕部位移；原生置信度不是外部标定的误差真值。

独立辅助清单human_training_inputs_v1/manifest.json含1743段：1582训练/161验证，832会话。DINOv2-L、当前本地V-JEPA2-L及OpenD4RT共5229教师记录，不是5229个视频。

正式辅助入口为examples.baselines.hypericl_pretrain.data_fixed_future_queries.ReferencesFixedFutureQueries。训练入口hypericl_pretrain.train_fixed_future_queries要求--reader-readiness对应human_aux_fixed_future_query_readiness_v2/report.json；本轮没有运行训练。

完整参考和观察前缀分别编码/缓存。未来分支只看截止时刻及以前；固定输入0.2/0.5/1/2秒请求，片尾实际截短时长仅作标签metadata，mask只参与loss。旧References的片尾query泄漏不用于正式接口。3486次双前缀及逐字段因果检查已通过，教师和目标缓存复用。

新增22参考为碗3、书5、遥控器11、柜门3。前19窗口中10个有1459帧右腕标签、9个仅视觉；柜门另有198帧右腕标签。最终桥分别在with_start_state_aux_data_handoff_v1/added19_same_reference与cabinet3_same_reference。SameReferenceAuxBridge按实际reference_id读同一窗口，不替换参考，也不自动给原750示范补新标签。

柜门仅批准抽象开右门并松手，exact_human_robot_pair_approved=false；没有逐帧动作、力或左右手到机械臂标签。遥控器8训练/3验证窗口分别来自各自一个会话，同会话intra的pair_valid=false；test暂无合法新增人类参考。杯/瓶图库只说明共享搬运技能，不是新选择程序的批准正参考。

1765辅助用途去重为1757完整源视频、837会话；原基准810人类参考另外保留。缓存和读取就绪不等于辅助loss已接入当前CNN，训练集成需后续明确实验。

## 采样建议和epoch

```python
from torch.utils.data import DataLoader
from examples.baselines.hypericl_data_factory.policy_training_inputs import (
    FactoryPolicyTrainingDataset, OriginalMajorityFrameBatchSampler,
)

data = FactoryPolicyTrainingDataset(manifest, ZQConfig(frames=32))
sampler = OriginalMajorityFrameBatchSampler(data, batch_size=16,
                                          original_fraction=.75, seed=42)
loader = DataLoader(data, batch_sampler=sampler, num_workers=0)
batch = next(iter(loader))
```

完整批默认12原始query＋4新增query；75%是query数量比例。新增按task family→goal→独立group→episode→native frame均衡。比例是初始建议，未证明收益。

epoch按原始1033711训练query帧完整shuffle一次，B16共86143批；尾批由sampler处理。新增2497训练轨迹、943118原生query帧。现有检查为索引预览与实际batch，没有跑完整epoch，尚无DDP分片。

Q2有效action query保留，但reference_pool_eligible=false者禁止作参考。机器人Q1与人类相机Q1/Q2是不同标准。每任务Q2上限、相机层及source-domain配额未实现，见quality_sampling_recommendation_v2。

碗45110015保持冻结正式accepted，但native连续10步成功为false；可选连续0.5秒严格池建议排除，尚未应用。Q1也不保证每帧无遮挡，93720004有13/771投影越界帧。

## 划分、语义和复现

最终物理初态、有限姿态分箱、原生命令三类跨split冲突0。1553个实测layout/pose-bin与3337套命令分开计数。同split多目标可共享布局，零冲突不代表所有近似重复和语义重合被排除。

原基准数据、Seen/Unseen列及评估协议保持可追溯。新增H4 L0–L2相同抽象目标已覆盖，35/75mm平顶支撑只部分涉及L3初态，不等于真实倚靠。1743辅助及历史预训练语义未全查，不能作全局严格零样本声明。

实际命令见startstate_extended_actual_invocations_v1/REPORT.md；详细、精确尝试分母和划分检查分别见data_preparation_multitask_report_v10_with_start_state_aux、core_plus_startstate3508_population_audit_v1、measured_split_overlap_with_start_state_v1。生产命令见batch_generation_commands_v1/REPRODUCE.md，新批先小批验证，不重跑历史seed填数量。

[更新前历史DATA_USAGE](/mnt/sdf/datasets/hypericl_data_factory_20261016_v1/final_goal_delivery_v1/history/DATA_USAGE.md)。旧冻结证据保留；本轮没有策略模型前向、优化器步骤或新的成功率结果。
