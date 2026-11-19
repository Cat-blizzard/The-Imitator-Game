# 自动关键状态与定位候选：SGP（State-grounded Proposals）

本模块把“人工写隐藏物体要求、挑关键帧、圈把手/背景 ROI”改为“可复算候选 → VLM 局部修正 → 下游追踪检查”。名称是方便描述模块用途，不代表已经验证的论文创新或通用能力。

实现：

- `r2h_pipeline/scene_proposals.py`
- `scripts/propose_scene_conditions_20261020d.py`
- `tests/test_scene_proposals.py`

实际 CPU 结果在 `/mnt/sdg/datasets/real_sim_human_pipeline/automation_20261020d/scene_proposals/`。建议使用 `*_v2` 目录；`*_v1` 是保留的初始候选版本。原始采集和生成视频未修改，没有启动训练。

## 为什么不必从提示词猜隐藏物体

现有仿真采集已输出真实可用的 QA 数据：

- `qa_visibility.npz`：对象分割可见像素、投影轮廓框、可见框、相机内外参。
- `arrays.npz`：各时刻对象位姿、实际 TCP 位姿、抓取状态、时间戳。
- `receipt.json`：资产 UID/名称以及实际把手网格初始世界坐标包围盒。
- 柜门 `joint_contact_measurements.npz`：仿真实际接触力、门角度等。

投影、分割计数的生成代码是 `PolicyGen/examples/baselines/hypericl_data_factory/rich_runtime_v2.py` 中 `VisibilityRecorder`。其对象顺序与 `unwrapped.objects` 一致，资产名称由 receipt 中对应顺序关联。这里不是读取人工标记的“柜子里有方块”来推断。

隐藏状态规则：物体初始投影在画面内，但实例分割像素数为 0；后续连续 3 帧超过 `max(4, 2% × 全片最大可见像素数)`，则建议加入显露状态视觉条件。阈值是工程候选阈值，未统计校准。不能因为零像素就声称具体遮挡者一定是柜门；代码输出 `occlusion_candidate`，不推断真实人类片段的隐藏物体。

柜门实际自动结果：方块初始 0 像素，首个持续显露时刻为第 276 帧（13.8 秒），第 330 帧达到最大 376 像素；梨始终可见，不产生隐藏物体要求。自动关键帧为 **0、202、276、330、364、498**。其中 202/364 来自仿真接触测量阈值变化，276/330 来自分割显露，而非 controller stage 的字符串名称。

这些是仿真时间戳，不能按时间比例直接改成人类生成视频的真实接触标注。

## ROI 怎么从几何得到，再怎么跨图像转移

1. 将 receipt 记录的初始把手网格世界坐标 AABB 的 8 个角点，用保存的 `camera_from_world` 与 `intrinsic_512` 投到仿真首帧。
2. 对仿真初始图与人类生成视频首帧做 SIFT 特征匹配、Lowe 0.72 筛选、4 像素 RANSAC 单应矩阵。
3. 几何支持不足时直接返回 unknown。足够时把源 ROI 转到目标图，并加少量边距。
4. 保存匹配数量、内点比例、重投影误差、内点覆盖范围、ROI 附近支持距离。这些是几何测量，不是概率置信度。
5. 背景区域独立由仿真首/末与两个中间帧的像素稳定性、角点数量提出。稳定区域仍可能不是最合适的语义背景，因此交给 VLM 确认。

新视频不是精确三维渲染，单应变换不保证所有深度一致。即使全局映射很好，局部把手可能被生成模型变形。因此所有转移 ROI 都保留 `semantic_identity_verified=false`、`requires_vlm_correction=true`。

自动建议的演员信息来自实际 TCP 位移：超过 3cm 的活动机械臂数量可以提示需要几只活动手；它不能推断人物数，也不能证明静止支撑手可删除。模块明确不自动把“一条活动臂”变成“画面最多只能一只手”。

## 已运行的四例

| 输入/目标视频 | 自动隐藏要求 | 关键帧数 | 几何转移结果 |
|---|---:|---:|---|
| 柜门仿真 → 旧人类视频 | 1 | 6 | 75/79 内点，中位误差 0.51px，候选 |
| 同仿真 → 新首尾条件视频 | 1 | 6 | 122/132 内点，中位误差 0.40px，候选 |
| 苹果搬运 → 人类视频 | 0 | 4 | 7/21 内点，比例 0.333，低于门槛，unknown |
| 书本放盒 → 人类视频 | 0 | 4 | 39 内点，中位误差 0.56px，候选 |

这是三个任务、四段目标视频的工程验证，不是自动定位准确率评测。没有阅读历史人工 ROI 作为输入。柜门候选叠图实际落在右把手、梨与右上背景，尚需 VLM 和后续跟踪验证。苹果是必要的失败路由示例：不把错误单应矩阵强行用于后续检查。

额外下游检查：直接把新柜门的纯自动把手/背景候选（未经过 VLM 修改）输入已有 `contact_diagnostics`。193 帧全部有跟踪支持，初始把手 28 个点、背景 40 个点，CPU 约 11.1 秒。产物为 `cabinet_new_v2/automatic_roi_contact_probe/`。这证明自动候选已可接到检查模块，并不证明全部点始终属于真正把手；框内邻近门纹理、手部遮挡依然可能导致身份偏移。所有物理和训练放行标志仍为 false。

## VLM 是可执行修正模块

每次离线运行同时生成 `vlm_correction_request.json`，包含：

- 原始 proposal 的内容哈希；
- 目标首帧文件哈希；
- 源/目标图像；
- 源几何框、候选目标框、资产名称与不确定性；
- 每个 ROI 需要 `keep / correct / unknown` 的明确接口。

修正使用一次已授权 Gemini 调用；显式 `--review` 才会调用，离线生成候选不会付费。调用结果用现有 `paid_stage` 缓存，避免重复提交。修正必须逐个 ROI 决策且不允许重复 ID，必须绑定当前图像和原 proposal。原 proposal 不会被覆盖。

输出：

- `proposal.json`：原始几何与视觉算法候选；
- `vlm_correction_request.json`：可检查请求；
- `vlm_correction_v1/`：原始 API 记录；
- `correction_patch.json`：仅包含 ROI 修改与理由；
- `corrected_rois.json`：供后续追踪使用的 VLM 审核候选。

VLM 的 keep/correct 只是二维定位审核，不会把 `physical_contact_approved` 或 `training_admitted` 改为 true。下一步仍需把 ROI 放入 `contact_diagnostics`，检查轨迹有效点数、跟踪丢失和关键接触时段，而不是凭单张框放行视频。

```bash
PolicyGen/.venv/bin/python data_generation_methods/scripts/propose_scene_conditions_20261020d.py \
  --input /path/to/existing/job/input.json \
  --target-video /path/to/generated_human.mp4 \
  --output /path/to/new/proposal_directory

# 单次缓存 VLM 修正；不产生新视频，不训练
PolicyGen/.venv/bin/python data_generation_methods/scripts/propose_scene_conditions_20261020d.py \
  --output /path/to/new/proposal_directory --review
```

## 尚未解决与后续出口

- 仿真中没有导出相机或对象实例信息时，不能精确复算隐藏物体和投影框，应让新采集接口强制保存这些 QA 字段；老数据只能走视觉候选/VLM，不伪造物理标签。
- 当前自动检测“隐藏后显露”，尚未覆盖所有任务逻辑，例如必须先取出再放回、保持容器内部为空、途中不可碰触某物。需要由人类视频事件规范及仿真状态谓词进一步约束。
- 背景 ROI 目前是固定画面候选。若相机移动，须加入相机运动估计/分区背景与显式失败状态。
- 没有使用 SAM2/CoTracker，也不声称这些模型已集成。未来可在 VLM 给出框后由分割跟踪保持实例身份，比纯 KLT 更适合遮挡恢复，但恢复必须重新确认身份。
- 三个任务还不支持宣称超过 Zero WAM 等工作的整体效果。应做固定任务与生成预算下的“纯提示词/几何候选/候选+VLM修正/完整事件闭环”对照，报告隐藏物体保真、额外肢体、接触可见性、每条合格样本成本与人工改动次数。
