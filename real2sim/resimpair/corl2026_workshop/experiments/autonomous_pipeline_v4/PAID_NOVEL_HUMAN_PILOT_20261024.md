# 新机制 Sim→人类视频付费试验 · 2026-10-24

用户已明确长期授权为任务数据生成自主调用 API，不再逐轮询问许可。本轮使用原生验收合格的已有轨迹，实际完成有限批次；不新增物理执行，不使用训练 GPU，不修改旧失败和数据。

实验根：`/mnt/sdg/datasets/real_sim_human_pipeline/novel_human_pilot_20261024_v1`。观看入口：[逐任务画廊](/mnt/sdg/datasets/real_sim_human_pipeline/novel_human_pilot_20261024_v1/gallery_v2/index.html)。机器可读证据：[delivery.json](/mnt/sdg/datasets/real_sim_human_pipeline/novel_human_pilot_20261024_v1/gallery_v2/delivery.json)。

## 范围与预算

预先固定按钮、钩取、搅动 3 场景，每场景 2 任务，共 6 个候选；每场景共享一张首图。相同来源与物理执行的派生内容没有拆到不同数据划分。

本轮配置模型：首图 `gemini-nano-banana-2.1`，视频 `doubao-seedance-2-5-260628`，审核 `gemini-3.1-pro-preview`。模型别名与服务商返回的 modelVersion 分开留存于请求／响应记录。

新增调用上限 image 3 / video 6 / review 9，并发 1，无语义重抽。原谱系 image 21 / video 13 / review 24 原样保留。真人来源窗口尚未覆盖这些新机制，因此这是未配对的 Sim→人类生成质量实验；正式 Real→Sim→Human 配对为 0，未准入训练。

## 实际结果

| 任务 | 冻结流程结果 | 失败／未知／后验争议 |
|---|---|---|
| 按钮：右手／红色 | 视频生成；审核已扣费但输出丢失，质量 unknown | video_review_result_unavailable |
| 按钮：左手／蓝色 | 视频生成；冻结审核未通过 | final_goal、event:plunger_1.pressed、event:plunger_1.rebound、goal:plunger_1.goal、hand:event:plunger_1.pressed、hand:event:plunger_1.rebound、joint_motion_between_endpoints、后验争议：possible_false_negative |
| 钩取：右手／红块 | 49 帧可见范围审核通过 | 见原结果与局限 |
| 钩取：右手／蓝块 | 49 帧可见范围审核通过 | 见原结果与局限 |
| 搅动：画面逆时针一圈 | 视频生成；冻结审核未通过 | cyclic_motion、后验争议：direction_observation_conflict |
| 搅动：画面顺时针一圈 | 49 帧可见范围审核通过 | 后验争议：possible_false_positive_exact_turn_count |

实际下载人类视频 **6 条**。自动状态统计：`{"review_output_lost_after_billed_completion": 1, "generated_video_rejected": 2, "sampled_visual_pass": 3}`。`sampled_visual_pass` 只代表指定抽帧中的模型观察，不能代替全部帧／真实 3D 物理标签或独立人类评估。

自动通过中 **1 条**因后验发现重复圈数而暂缓晋级；没有发现同类冲突的采样通过 **2 条**，均为钩取。原判定保留在原目录，争议另存 diagnostics/review_disputes.json。不能把三条 VLM 放行都算成可靠有效样本。

直接查看诊断：[蓝按钮局部图](/mnt/sdg/datasets/real_sim_human_pipeline/novel_human_pilot_20261024_v1/diagnostics/blue_cap_crops.jpg)、[逆时针任务的端点叠加图](/mnt/sdg/datasets/real_sim_human_pipeline/novel_human_pilot_20261024_v1/diagnostics/stir_counterclockwise_pixels/tip_overlay_49.jpg)、[顺时针任务的端点叠加图](/mnt/sdg/datasets/real_sim_human_pipeline/novel_human_pilot_20261024_v1/diagnostics/stir_clockwise_pixels/tip_overlay_49.jpg)。这些是原视频的后验局部观察，未参与生成或覆盖原审核。

## 计费与耗时

本轮已开始调用：`{"video": 6, "review": 9, "image": 3}`；累计谱系：`{"video": 19, "review": 33, "image": 24}`。失败与输出丢失的调用全部计入，旧消耗没有清零。

服务商消费记录合计 **15644837 额度单位**，按当前公开 `quota_per_unit=500000` 名义折合 **$31.289674**。分阶段额度快照合计 **15644837**；两者是否一致：**True**。

这是服务商 API 额度消耗，人民币现金实付仍未核实。价格来自本次实际消费记录和公开额度单位，未用 token 数或模型网页基础单价捏造视频费用。模型／时间／扣额可与本地串行调用对应，但没有逐请求响应 ID 的完整对账；共享 token 的其他外部调用仍需区分。

视频提交先扣额度，完成后还有调整，已分别保存。只看提交返回的 usage 或基础报价会漏记实际费用。旧账本零额度不代表服务商余额为零。

整批墙钟 **83.34 分钟**，含响应超时、本地诊断及冻结版本恢复间隔，不能视为纯模型流水线吞吐。

| 阶段 | n | 最短／中位／最长（秒） |
|---|---:|---|
| image | 3 | 37.5 / 50.8 / 53.2 |
| anchor_review | 3 | 34.2 / 34.3 / 59.4 |
| video | 6 | 259.4 / 278.7 / 592.1 |
| video_review_49_frames | 6 | 162.6 / 247.5 / 615.6 |

上述串行 API 阶段累计 **68.12 分钟**（包括一次审核超时）；与 83.34 分钟墙钟的差额包含诊断、版本冻结、恢复和其他本地开销，不能全归到模型推理。

| 阶段 | 已开始调用 | 实测额度单位 | 名义 USD |
|---|---:|---:|---:|
| image | 3 | 150138 | 0.300276 |
| video | 6 | 15028578 | 30.057156 |
| review | 9 | 466121 | 0.932242 |

image/review 的此表包括该 API 阶段的额度查询；video 从提交计到首次观测服务端终态。各条提交到下载、HTTP 传输和流块数另见 delivery 的 stages；排队和模型计算时间没有被虚构拆分。正式合格配对数为 0，因此每条正式有效配对费用不可计算。

## 实现与验证

- `mechanism_conditions.py` 从既有原生场景和事件谓词自动导出颜色、功能构件、中间空间点、关节状态证据和画面中的循环方向／圈数。保留原物理合同与真人来源缺失状态，不逐样本手写修复提示。
- `paid_pilot.py` 复用既有图像／视频 API、人数／手归属审核和媒体时钟验证。三个共享首图、六个任务输入在付费前冻结；一次原生执行不计作多条新物理轨迹。
- `mechanism_review.py` 增加功能几何、关节中间运动及周期方向检查，未删减原人数、对象、手、顺序或计数规则。
- `billing.MeteredBudget` 保留累计调用预算，新增调用前后额度快照、异步视频终态核对及服务商逐笔消费日志。凭据只在私有配置／内存中使用，没有写进报告、提示或日志。
- `human_r1` 在本地发现共享首图同像素异路径的不可变绑定冲突，付费 0；`human_r2` 修正并运行。`human_r3` 仅为未开始任务切换审核传输及账单恢复，不重生成／重审首个终态任务。
- `human_r4` 补 `image_response.py`：同一服务商可返回 Markdown CDN 图片链接，而非 inlineData。对已有完整响应执行一次无凭据的媒体 GET、限制大小并完整解码；保存原响应、失败与下载字节绑定，再恢复原调用结果。没有新增图像 POST 或重抽。批次异常退出会更新 stopped 状态，避免进度文件假报 running。
- `human_r4` 的 49 个模块与当前开发树一致，完整本地回归 **117 项通过**，13 个测试源文件另有冻结副本。检查涵盖基础流程、空间／圈数、同图异路径、额度未知、响应丢失和媒体格式恢复。原生双视角全部解码并核对状态时间轴；只读交付审计复查 6 例和 18 次付费尝试，未修改视觉规则。见 verification/final_binding_audit.json 与 full_human_r4_tests.log。

本轮没有逐样本手写修复提示或手工补语义标签，但并非无人干预的连续生产：agent 在付费前修正一次共享首图绑定实现问题，运行中进行两次协议恢复（审核输出丢失、CDN 图像格式），并执行了后验画面诊断。这些工程介入不能从总耗时或人工介入统计中抹去。实际输出媒体尺寸、完整解码数和帧率逐条保存在 delivery.json。

## 实际问题、处理和未解决项

1. **已扣费但同步审核响应丢失。** 第一条红按钮审核在服务商日志中完成并扣费，本地等到超时仍无内容。保留原 unknown 失败、额度和视频；以唯一模型／时间／精确扣额记录登记当前输出不可用，不补造结果、不重复提交。其他尚未开始任务使用既有流式通道，实际块持久落盘。此轮证明流式通道可返回完整观察，不证明它因果上加快模型推理，也未证明不再丢响应。
2. **全图 VLM 可能漏看小幅动作。** 蓝按钮审核称没有下压／回弹，但后验局部裁剪显示按钮从凸起到压低、再恢复。保存为审核误判疑例，原审核仍是未通过。颜色区域／边界跟踪可廉价辅助诊断，但受遮挡影响，不能充当真实行程或接触力标签。见 diagnostics/blue_button_review_disagreement.json。
3. **中间过程约束仍未被可靠执行。** 按钮和顺／逆搅动可能有相同起止状态；只有事件名或两张端点图不足以限定中间过程。本轮已明确编译方向、圈数和原生路径点，但逆时针一圈视频仍反复搅动，冻结审核也以圈数不符拒绝。方向上 VLM 称逆时针，后验画面和局部颜色端点跟踪则支持顺时针，另存为审核争议。固定颜色工具得到约 1.8 圈的二维估计，但有小球颜色混入与遮挡，未晋级为通用验收。证据在 diagnostics/stir_counterclockwise_pixels/；原判定不覆盖。
4. **真实来源和泛化范围不足。** 新机制是工程探针，未绑定新真人窗口；没有新真实家庭扫描资产。本轮固定布局的证据不能外推任意任务、杂乱遮挡或新本体。严格物理库另外仍有平衡梁回零、双球组合两个分支未解决。

5. **接口与运行状态也会误报失败或在跑。** 搅动首图原响应实际含 CDN 图片，旧解析器却报缺图；原 worker 异常后 progress 仍是 running。此次恢复真实图片并在候选中补异常状态更新。原始失败、停止进度快照和恢复记录均保留，不能把恢复后的完成当作最初就顺畅完成。

6. **同一圈数审核仍可漏检。** 顺时针一圈视频被 VLM 放行，理由仅确认做过一个完整圆周，没有排除后续重复。后验 49 帧叠加图和不变颜色规则的二维端点估计约 2.9 圈，支持过量重复的疑点。它已单独标记为 disputed_visual_pass_held_from_promotion；没有修改旧 pass，也没有把该启发式工具晋级为默认验收。

## 决策与下一步

保留 API 计费／恢复改进和所有样本，不以 VLM 通过数宣布数据管线整体晋级。优先把原生功能区域投影接到自动局部检查，使用无动作、错误对象、错误方向等反例校准误报与漏报；在同一新审核协议下重比较既有结果。对闭环动作，下一候选需验证中间关键状态／分段条件，而非继续依赖相同起终态加长文本。之后再扩展未参与调试的布局、真人来源与资产。内容失败不在原运行里补 prompt 重抽。

接口参考：[服务商报价与账户入口](https://api.echoflowai.cc/pricing)、[兼容日志接口文档](https://doc.newapi.pro/api/fei-log/)。实际收费依据以本实验 billing/ 的脱敏服务商记录为准。
