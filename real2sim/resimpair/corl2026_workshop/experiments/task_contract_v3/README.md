# 多机制任务合同与人类视频候选

这是冻结管线外的 opt-in 候选，复用 2026-10-22 五任务的已完成执行。原 619 文件快照、物理轨迹、失败和苹果视频不改。当前实际进度以 SDG 运行目录的 `PROGRESS.json` 和累计调用账本为准。

运行目录：`/mnt/sdg/datasets/real_sim_human_pipeline/five_task_probe_20261022_v1/human_generation_v3/`。

2026-10-23：**五个人类视频均已实际生成、下载和审核**。[对照播放页](../five_task_probe_20261022/results/human_generation_v3/index.html) 与 [完整报告](../../notes/FIVE_TASK_HUMAN_PIPELINE_20261023.md) 保留全部结果。苹果、柜门、多米诺有任务顺序支持；堆叠的三层终态正确但对象顺序为 2→1→0；插接的左右手支撑正确但插头变成薄片、孔区失真。后二者按当前合同拒绝，正式配对及训练准入仍为 0。

本候选修复原统一入口只接受严格苹果的断点。`compiler.py` 从原始状态、抓持、接触和关节测量编译初态、事件、终态、对象角色、逐手状态和持续支撑约束；`prepare_probe.py` 将四条已完成轨迹与明确记录的来源观察及扩增映射接入同一接口。苹果保留已经完成的旧入口结果。

来源合同由已审核观察和显式映射编译。柜门复用完整原子任务窗口；多米诺声明计数／布局扩增；堆叠声明子技能组合；刚性插接声明机械类比及终态延伸。编译成功不会自动升级成精确真人配对。当前来源观察由本会话直接看图及历史绑定审核提供，尚不是一个独立完成未知视频理解的自动视觉模型。

编译器保留全部堆叠原子事件，使用事件 DAG 和持续约束，拒绝控制器阶段名冒充接触时间。插接的终态明确为左手保持接收件、右手释放插头，持续持握检查覆盖 plug grasp 到 terminal 的 402 帧。首图、终图和视频 prompt 都由同一合同生成。实际模型已接受 8 秒与 12 秒请求，输出分别为 193／289 帧、24 fps，约 8.04／12.04 秒；解码成功和视觉任务成功分别记录。

标准首尾图模式要求两张条件图均通过原 `single_actor_v2.1` 六个人物检查及新增的多机制可见状态候选检查。后者保留任务、身份、比例、角色、状态、非目标对象和相机七个维度，增加逐手持物状态。它不是旧苹果静态门的替换，也不是自动视觉检测器；不可见身份保持未认证。来源、仿真、静态图、动态视频和最终配对资格分别记录。

插接终图两次修复仍看不清第二个空插孔，端点门保持 blocked。`diagnostic_execution.py` 新增显式登记的单首图诊断条件：保留合格首图和完整事件／目标关系，不发送失败终图。它需要绑定具体失败、原审核和实验假设，不自动回退、不修改原端点门、不放行坏首图。该诊断视频已真实生成并因几何失真被拒绝。四条其他视频仍使用已审首尾图。后续重入先恢复已经开始的视频，不重发新 POST。

常用入口（在工作区根目录运行）：

```bash
PolicyGen/.venv/bin/python -B corl2026_workshop/experiments/task_contract_v3/run.py check --task /absolute/task/directory
PolicyGen/.venv/bin/python -B corl2026_workshop/experiments/task_contract_v3/run.py step --task /absolute/task/directory
PolicyGen/.venv/bin/python -B corl2026_workshop/experiments/task_contract_v3/run.py step --task /absolute/task/directory --execute
PolicyGen/.venv/bin/python -B -m corl2026_workshop.experiments.task_contract_v3.preflight
```

`step` 每次推进一个可恢复阶段；需要直接视觉观察时输出审核包，未知和失败不会自动变成通过。`record_observation.py` 和 `record_video_observation.py` 只把已经完成的逐项观察写成审核文件，没有默认 pass。`media.py` 完整解码实际生成视频并生成带帧号的采样图，不自动判断视频成功。CLI 支持 `review-image` 和 `review-video`，均要求明确提供已完成的直接观察文件。

原始不可变预算为 image 12／video 6／review 0；同一账本的 `allowance_amendment_01.json` 显式将图像累计上限改为 16，视频上限仍为 6。并发 1，单次 600 秒。本批已使用 image 16／video 6：获得 12 张新图和 4 条新视频，4 次图像与 2 次视频断连仍保留 provider outcome unknown。已完成苹果的 image 2／video 1 通过 `authorization_and_lineage.json` 累加，总计 image 18／video 7。上限和全部计数保留，不清零、不换账本；本批已完成，勿重发已生成的五条。实扣货币费用未核对，次数不等于服务商账单记录数。

当前 CLI 通过 `diagnostic_execution.py` 分派，标准图像与视频路径分别复用 `image_execution_direct.py`、`video_execution_direct.py`。图像使用相同配置模型的原生流式响应；视频复用既有 Seedance 客户端。直连传输不继承环境代理，保留 TLS 验证、禁止重定向和不自动重发。原执行器字节和历史请求保留，已被 provenance 绑定的源码不能原地修改。

`budget_recovery_v4.py` 保持同一注册表、同一账本和显式修订后的累计上限。同步图片结果无法恢复时，须完成只读核对、完整超时窗口和 reconciliation 才能退休本地尝试；同一阶段禁止再提交。视频有 task ID 时只查询原任务；无 ID 时由 `video_recovery.py` 要求超时后的两次完整任务列表核对，并在每个后续 POST 前检查晚到任务。原未知调用始终计数，不伪造 `result.json`，列表缺失不证明没有执行或收费。当前无未处理未知提交、无活动视频任务。

软件测试覆盖真实四任务编译、12 个堆叠原子事件、持续支撑、错误接触时序、缺失抓持、来源越界／串源、首尾图调换、替换条件图、无观察 pass 标签、累计预算、任务列表恢复、直连单 POST 和不完整流式响应。本轮新增四个诊断条件检查全部通过，原 47 项结果保留。测试和直接视觉观察均不冒充跨任务泛化实验。

堆叠顺序的关系描述修复已保存到 `04_jenga_stack/video_02/repair_brief.json` 和 `prepared_video_request.json`，未提交新生成。当前最大内容缺口是模型对事件身份和细小接口几何的遵从性。未来应在预先固定的合同下比较修复，并明确同形对象是否允许交换，不能对观察到的失败事后放宽规则。
