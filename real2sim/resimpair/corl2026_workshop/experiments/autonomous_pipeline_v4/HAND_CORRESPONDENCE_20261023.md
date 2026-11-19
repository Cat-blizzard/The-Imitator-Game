# 2026-10-23：仿真—人类参考的操作手对应修复

已修复共享代码中的手部映射错误，并加入生成前检查和版本化审核。**本地验证通过；尚未重新生成人类视频，不能宣称视觉问题已解决。** 候选为 `hand_correspondence_r1`，不替换历史 release，不修改原合同、原视频或原拒绝结论。

## 真实失败与首个错误环节

用户给出的 `mechanism_workbench_C__passive_chain/workbench_r2/human.mp4` 是多米诺视频。插座案例实际位于同批的 [`mechanism_workbench_C__insertion/workbench_r2/human.mp4`](/mnt/sdg/datasets/real_sim_human_pipeline/autonomous_mechanisms_20261023_v5/runs/mechanism_workbench_C__insertion/workbench_r2/human.mp4)。两条都已核对。

[插接对照帧](/mnt/sdg/datasets/real_sim_human_pipeline/hand_correspondence_20261023_v1/insertion_comparison.jpg) 显示：人类 f57（2.38 s）从画面右肩伸出的解剖学左手跨身拿起画面左侧接收件，随后持续持握；右手取插头并插入。仿真则是画面左侧基座的 `arm0` 持握接收件，`arm1` 插入。对照图按各视频相对进度取样，不是事件逐帧对齐。

原生 `robot_roots_world` 给出两基座约为 `[-0.60,-0.35,0]`、`[-0.60,+0.35,0]`，朝向一致。当前双 Panda 的前方为局部 +X，上方 +Z，身体右方为 `forward × up = -Y`。因此应为 **arm0→人类右手，arm1→人类左手**。对面相机里，人的右肩在画面左侧，不能把画面左右当成人体左右。

旧 `workbench.protocol` 和通用插接 builder 硬写了 `arm0→left、arm1→right`，任务意图也写成 `left_holds_receiver_right_inserts`。生成视频与这个错误分工一致：首个已证实错误在任务条件编译之前，不足以归因于视频模型不服从条件。r4 的旧 VLM 还把视频中的左右手判断反了；原拒绝保留，但它的理由不作为正确人体侧别证据。

## 实现与失败行为

1. `hand_correspondence.py` 从原生基座坐标系和 grasp/contact 通道绑定推导每个 effector 的解剖学手侧。机器人数组索引、对象左右位置和当前腕部位置均不充当人体侧别。相机只确定初始肩部的画面顺序；不虚构人类三维关节标签。
2. 规划端工作台及搬运 builder 根据场景几何声明手侧；没有布局输入的通用插接 builder 不再硬编码，必须在编译端用实际测量确定。编译端独立重算；声明冲突、缺少测量、混用机器人通道、反射坐标系、对置/前后排列基座或侧向投影不明确时，直接停止，不提交生成。当前自动映射仅适用于同向并排的 Panda 基座约定。
3. 初图、视频事件、持续持握和终态共用一个绑定。初始空手不再意味着后续可以任选手。手跨过身体中线仍保留身份；允许人体采用自然运动路径。旧 v4 搬运编译器丢弃 `arm_index` 的缺口也补齐。
4. `review.py` 与独立的 `hand_review.py` 增加 `resimpair_anatomical_hand_review_v1`。首图检查身体朝向；每个有主动操作手的事件及终态持握单列审核。模型还须输出 `hand_traces`：实际手侧、初始肩部画面侧、肢体追踪状态、初态与事件帧引用及具体观察。确定性门禁拒绝错误、未知、遮挡无法确认、缺项、伪造帧和与事件证据不对应的结论；总体全 yes 不能抵消错误手侧。

VLM 仍可能同时看错手侧和肩部来源。这是更明确的条件与证据约束，不是已经验证的像素级人体追踪器，也不保证未采样帧。新 wire 字段尚未经远程模型验证；不将本地测试当成视觉检出率。

## 实际验证

| 验证 | 结果与范围 |
|---|---|
| 软件回归 | **55/55**，其中新增 20 项；含镜向观察、机器人索引交换、整体坐标旋转、跨身动作、未知/错误手审核与真实工作台记录 |
| 原插接完整编译入口 | 在原物理资格通过后，拒绝 `hand_mapping_conflict:arm0:declared_left:measured_right`，不触发生成 |
| 工作台 2 条原生记录 | 新共享 builder 导出的所有物理事件、帧号、谓词结果与原始结果完全一致；只修正操作手条件。795 帧插接、426 帧多米诺均为历史回放，不是新执行 |
| 同初态复用 | 两种任务仍得到相同首图条件；测量 hash 留在证据里，不把每次执行细微数值差异变成重复首图请求 |
| 其他保留记录 | 9 条多布局搬运 + 旧柜门/多米诺/插接，共 **12 条**兼容性回放；原 **7 个物理反例**仍全部拒绝。这是回顾性兼容性，不是留出视觉验证 |
| 真实视频审核接线 | 两条原视频各准备 49 帧请求；插接 5 项手部追踪、多米诺 1 项，全部绑定冻结候选代码；**未提交**模型 |
| 新开销 | 新 API **0**，新仿真 **0**，新训练 **0**。数值映射为本地常量级工作；审核仍使用同一次调用，但增加文本/输出，实际时延、货币费用未知 |

全部产物在 [`/mnt/sdg/datasets/real_sim_human_pipeline/hand_correspondence_20261023_v1/`](/mnt/sdg/datasets/real_sim_human_pipeline/hand_correspondence_20261023_v1)。主要入口为 `workbench_replay/audit.json`、`compatibility_replay/audit.json`、`verification/tests.log`、`verification/review_preparation.json`、`releases/hand_correspondence_r1/release.json`。修改前源码、文档及两份状态均有备份。

## 决策与下一步

保留为候选：几何绑定 bug 已由代码与真实测量复现并修复，视觉收益未验证。新条件和新审核同时改变，不能把后续收益全部归因于 prompt 或单一模块。无新合格配对；既有通过率、成本分母、论文实验计数不变。

预算重新核对：当前批 image1/video2/review12，谱系 image21/video13/review24，剩余 image2/video1/**review0**。不绕过审核提交视频，不因新目录或 release 重试旧终态。没有申请或配置新增付费预算。

下一次有明确累计预算的新实验，应先冻结同侧别规则和独立正负视觉样本，检查新审核能否辨别肩—肘—腕的归属；再对新登记的插接布局和另一种交互机制做小批生成。两侧操作、跨身、遮挡都要覆盖。基线与候选使用同一新标准，分开记录生成服从率、审核漏检/误拒及 unknown。可消融“仅修正自动绑定”与“绑定＋结构化审核”，不得以审核分数替代像素证据。

只做本地回放的可复用命令（输出必须选新的审计目录）：

```bash
PYTHONPATH=/mnt/sdg/datasets/real_sim_human_pipeline/hand_correspondence_20261023_v1/releases/hand_correspondence_r1:/mnt/sdf/dataset/zhouxunzhe/zhiqi \
PolicyGen/.venv/bin/python -B -m autonomous_pipeline_v4 replay-hand-bindings \
  --root /mnt/sdg/datasets/real_sim_human_pipeline/autonomous_mechanisms_20261023_v5 \
  --output /mnt/sdg/datasets/real_sim_human_pipeline/hand_correspondence_20261023_v1/replay_next
```

命令不运行环境、不调用 API、不修改原记录；生成的 `replay_preview` 不能作为新自主执行合同投喂生产运行。
