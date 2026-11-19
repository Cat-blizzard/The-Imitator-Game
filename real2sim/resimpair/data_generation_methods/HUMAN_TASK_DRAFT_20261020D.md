# 真人视频任务草案：盲观察入口

新增入口：

- `r2h_pipeline/human_task_draft.py`
- `scripts/extract_human_task_draft_20261020d.py`
- `tests/test_human_task_draft.py`

用途是把**已选定的原始真人候选窗口**转换成带帧/PTS证据的任务草案。它还不是自动从全库发现任务，也不会把 VLM 描述直接变成仿真合格规范。

## 本轮真实准备

输入为真实 EgoDex 柜门原子窗口：

`/mnt/sdf/datasets/hypericl_data_factory_20261016_v1/cabinet_atomic_reference_crops_v2/ego_part2_insert_remove_furniture_bench_cabinet_1766/atomic_native.mp4`

这段窗口是此前外部选定的72帧候选，不能把窗口的发现记作本模块成果。本次提取请求只给24张等间隔真实帧、实际PTS、窗口起止及源总帧数。没有提供历史手写任务描述、任务名称、旧审核结果、仿真状态或期望成功条件。API图片最长边1024，保持纵横比，JPEG92；原始帧和解码RGB哈希也保留。

产物：

`/mnt/sdg/datasets/real_sim_human_pipeline/automation_20261020d/human_task_drafts/cabinet_blind_v1/`

其中 `packet.json` 是原始帧与时钟绑定，`request.json` 可以直接查看模型到底被给了什么。准备阶段零API调用。

## 输出什么

VLM给出实体及固定身份、初始关系、最终关系、可见事件、观察到的顺序、窗口完整性候选、多手活动/支撑/静置角色、隐藏状态不确定性、局部补帧请求。

必须区分：

- `observed_order`：这段视频中看到的先后顺序。
- `necessary_precedence`：任务因果上必不可少的先后依赖。本模块始终写为 `unknown`、空边集；一段执行不能证明所有观察顺序都必要。
- `partial_action`：窗口可能只包含长任务的一部分，不能自动升级成完整任务。
- `support_or_stabilize`：静止支撑手可以被记录，不能因为动得少就删去。
- `hidden_state_observations`：后段显露是后段证据，不允许用后段画面直接断言初始帧可见的关系。

候选中 `manual_semantic_reviewed=false`，`human_sim_equivalence=unverified`，`physical_contact_verified=false`，`training_admitted=false`。

## 检查什么

本地验证器检查JSON结构、实体引用、唯一ID、引用必须属于实际提供帧且PTS完全对应、初始/最终关系的时间范围、可见事件必须有证据、顺序图无环且不能与证据区间相反、手角色必须指向hand实体、补帧请求必须由真实提供帧界定。

这些检查保证“模型没有引用没看过的帧”，不能保证图像理解本身正确。草案仍需独立审核或局部密采样复核，后续才能编译为仿真任务规范。

原始响应和验证错误全部保存；不自动重新提交。模型返回未知是有效输出，不能为了表格齐全而补造结论。

```bash
# 离线准备，视频窗口由调用方提供
PolicyGen/.venv/bin/python data_generation_methods/scripts/extract_human_task_draft_20261020d.py \
  --video /path/to/real_human_video.mp4 --start 0 --end 72 \
  --output /path/to/new/output

# 单次缓存提取；调用方明确启用时才使用API
PolicyGen/.venv/bin/python data_generation_methods/scripts/extract_human_task_draft_20261020d.py \
  --output /path/to/new/output --extract
```

成功时输出 `draft_task_contract.json`。这不是既有生产任务规范的自动替代，不修改旧manual审核标记，不自动启动仿真或训练。

## SGP ROI修正的失败处理补充

苹果候选的第一次实际VLM回复为梨/苹果给出了新框，但对几何未知的背景框错误返回 `keep`。验证器正确拒绝 `Cannot keep an unknown geometry proposal`。原始回复保留，`validation_error.json` 记录绑定和原因。

CLI现在提供显式 `--repair-review`，以同一批图片、原始回复和验证错误发起一次独立版本 `vlm_correction_repair_v2`。规则不放宽：未知框只能被 `correct` 重新定位或保持 `unknown`，不能被keep。它不会覆盖v1回复，重复执行只读取缓存。
