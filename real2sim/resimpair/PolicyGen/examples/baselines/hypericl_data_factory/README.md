# 按任务扩充训练数据

[当前数据、任务和视频索引](/mnt/sdf/datasets/hypericl_data_factory_20261016_v1/DATASET_INDEX.md) · [动作、时间和读取约定](DATA_USAGE.md)

最终数据整合、真实读取及划分检查已完成，没有启动模型训练或修改模型架构。全目标逐项交付结果见[54条验收报告](/mnt/sdf/datasets/hypericl_data_factory_20261016_v1/final_goal_delivery_v1/REPORT.md)。历史版本及其通过/失败记录保留，不用历史中间版本代替最终证据。

## 最终数据

- 3508次正式尝试：3337接受、171拒绝，包括原核心3460次和另批起始关节变化48次；起始姿态扩展的11次资格尝试排除。
- 新增机器人3337条、1286058原生帧：杯1240、瓶1182、碗257、书160、遥控器338、右柜门160。
- 最终清单7409行＝原4050行＋3337机器人＋22人类参考窗口。原行、归一化及旧v7全部7365行有序前缀保持。
- 新增机器人训练2497、验证268、测试572。原3000机器人训练、750人类训练参考、240机器人验证、60人类验证保留。
- 另有1743段独立人类运动/未来视觉辅助数据；连同新增参考用途去重为1757段完整源视频、837会话。窗口、用途、教师缓存和视频视角不能相加当独立示范。

最终清单：`/mnt/sdf/datasets/hypericl_data_factory_20261016_v1/policy_multitask_append_v8_with_robot_start_state/policy_manifest.json`。

SHA256：`5a7c5fe857e397acc6a7745679a9fbf9e21f18f3a17fc04f796694a62f9aee9c`。

## 直接读取

在PolicyGen目录运行：

```bash
.venv/bin/python -B /mnt/sdf/datasets/hypericl_data_factory_20261016_v1/data_preparation_multitask_report_v10_with_start_state_aux/read_batch.py --route human_to_robot
```

只读一个CPU batch，不创建模型或优化器。另支持robot_to_robot、same_embodiment_cross_scene以及--split train|validation|test。没有合法参考的路由明确不可用，不跨split挪用参考。独立人类辅助使用另一个读取器，见DATA_USAGE。

## 生产和检查

每类任务由场景配置、目标程序、阶段技能、IK/碰撞规划、PD控制、真实结果检查及同步记录组成；根据本次布局重新计算抓取与目标。正式数据均在双Panda场景由单活动臂完成，没有已验收双臂协作示范。

- [筛选、采集和检查命令](/mnt/sdf/datasets/hypericl_data_factory_20261016_v1/batch_generation_commands_v1/REPRODUCE.md)
- [最终整合实际命令、PID及退出记录](/mnt/sdf/datasets/hypericl_data_factory_20261016_v1/startstate_extended_actual_invocations_v1/REPORT.md)
- [详细数据报告](/mnt/sdf/datasets/hypericl_data_factory_20261016_v1/data_preparation_multitask_report_v10_with_start_state_aux/REPORT.md)
- [完整正式尝试统计](/mnt/sdf/datasets/hypericl_data_factory_20261016_v1/core_plus_startstate3508_population_audit_v1/REPORT.md)
- [质量与采样建议](/mnt/sdf/datasets/hypericl_data_factory_20261016_v1/quality_sampling_recommendation_v2/REPORT.md)
- [人类筛选阈值及限制](/mnt/sdf/datasets/hypericl_data_factory_20261016_v1/HUMAN_SCREENING.md)
- [投产资产来源与许可补证](/mnt/sdf/datasets/hypericl_data_factory_20261016_v1/production_asset_provenance_supplement_v1/REPORT.md)

新批使用新输出版本与种子，小批实际验收后再扩量；不要重启已终态原计划或重试物理失败来填成功数量。冻结采集器、资产、清单和旧报告保持不变。

初始采样建议原query75%、新增25%；新增按任务、目标、独立布局、录制、帧均衡，未证明训练收益。Q2有效action query保留，参考资格不足者不能作reference。质量/相机配额与DDP尚未实现。

三教师缓存、腕部及未来视觉读取已就绪，辅助loss尚未接入当前CNN。原Unseen列保留，但新增H4目标已覆盖，不能笼统称扩充后结果为严格未见任务泛化。

数据根目录：/mnt/sdf/datasets/hypericl_data_factory_20261016_v1/。EgoDex/HOT3D通过/mnt/sdc/datasets/迁移路径读取，没有复制新大库。

[更新前历史README](/mnt/sdf/datasets/hypericl_data_factory_20261016_v1/final_goal_delivery_v1/history/README.md) · [下一批优先级](/mnt/sdf/datasets/hypericl_data_factory_20261016_v1/next_batch_task_priorities_v3/REPORT.md)
