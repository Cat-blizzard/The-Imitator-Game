# 独立仓库接手入口

2026-11-01 最新：Pick-and-place v2 六步试点完成，见[报告](reports/pp2_pilot_20261101/README.md)。语义任务、经典资产、并行执行、语义验收和房间渲染已打通。最终量产批次 24 个场景、139 个任务中 132 个通过 v3 验收，反例 0/637。人类参考视频因服务商两次 503（额度 0、无任务 ID）而停止，生成视频的盲辨识率仍未测。本轮付费调用为图片 4、视频 2、审核 26 次，名义约 $0.59；无活动作业。入口：`docs/PICKPLACE_V2_SPEC.md`、`resimpair_pp2/`、`CURRENT_STATE.pp2_pilot_20261101`。下文为更早记录。

2026-10-31 接续更新（以 CURRENT_STATE.active_raw_asset_rollout_20261031 和实际 PID/锁为准）：原始资产主队列 22,344 来源已恢复，CPU/GPU 正在原账本续跑，不能重复启动。新冻结候选已接到真实 task consumer：**682 来源 / 10,708 个离线合格抓姿，198 接收容器来源 / 208 变体**；旧默认 935 库未改，二者有重叠且尺寸版本不同，不能相加。715 个输入中 33 个最终碰撞文件复载不一致被保留为失败。

XML 706 来源原生加工已全部终态，428 native-ready；固定4个cuDex首试全部导出0，不能将428记作可抓资产。2,859个新增容器来源已处理，240个几何/名称候选中98个适合作为深开口/浅盘接收候选，真正宽接收面仍不足。已修正最终碰撞文件未复查的问题：蘑菇32部件最终重载通过，同尺寸新缓存61候选，机器人成功尚未验证。

原资产接入轮12次实际机器人执行已封存，完整通过0，包含真实脱落、端点/完整路径失败和一例物理事件完成但碰撞几何不符。此结论不覆盖旧多物基线。新工作区约束和源优先采样已接线，但固定6物场景仍未找到全操作可行布局。没有新API、人类视频或正式配对。详见 [接续状态](reports/raw_asset_rollout_20261031/interruption_resume_01/CHECKPOINT.json) 和 [独立审计](reports/raw_asset_rollout_20261031/resume_independent_audit_r1/FINAL_REVIEW.json)。下文“无活动任务”等属于更早记录。

2026-10-31 最新：已完成[资产接入范围、尺寸与上方释放复盘](reports/asset_release_rethink_20261031/README.md)，证据见 [FINAL_EVIDENCE.json](reports/asset_release_rethink_20261031/FINAL_EVIDENCE.json)。935抓取来源只是旧5,524输入范围内的非空缓存子集，58,117原始来源没有全部加工。887种Objaverse中的779种继承了更早的夹爪归一化尺寸，相对父版本scale=1不能称为保留原始尺寸。

直接处理37个原尺寸GSO/YCB容器，新增7个独立几何候选；默认容器仍148来源，非默认 `runs/raw_container_candidate_bundle_20261031_v1/BUNDLE.json` 已被原任务工厂加载为155来源/166变体。尺寸覆盖候选实际编译2场景/8任务定义，尚未执行物理；四档等额压力测试的2/12不能当一般随机采样成功率。上方释放接口接通，一个同场景三物混合放置任务3/3操作通过原独立验收，未推广为所有任务成功。

本轮新增1次CPU物理，累计98/128全部终态、106已分配、8个旧准备任务未提交；37个容器CPU作业也全部终态。无活动作业、无新API/GPU/RGB/人类视频。后续优先补原始资产直接接入、大接收面和尺寸采样偏置；不要重启旧队列或重跑本轮37个来源。最新动态状态为 `CURRENT_STATE.active_asset_release_rethink_20261031`，下文“本轮”“最新”均保留其各阶段历史范围。

2026-10-31 场景接口更新已完成：[架构与使用](docs/SCENE_TOOLS_V1.md)、[真实对照与剩余问题](reports/architecture_20261031/README.md)、[冻结证据](reports/architecture_20261031/FINAL_EVIDENCE.json)。`scripts/scene_tools.sh` 提供10个场景查询、诊断、不可变修订与预算执行接口，57项测试通过，5类真实失败可回归。保留 cuDex / SAPIEN 后端，没有安装或替换成 SimScene。

本轮4次物理已结束：7物任务移动15mm落点后7/7独立通过，属于新布局；8物精确目标重排仍失败，定位为下放起点已偏12.1953mm，下一步是高位实测载物对齐。原9例统计和默认控制器不改。累计97/128物理全部终态，105已分配，原8任务仅准备未提交；API/GPU/新人类视频/完整配对新增0。本轮无活动作业。下方93/128及其他状态均保留其历史时间范围，最新状态见 CURRENT_STATE.active_architecture_20261031。

2026-10-31 本轮整合已结束：[实际结果与未解问题](reports/full_asset_integration_20261031/README.md)、[四场景首末帧图片](reports/full_asset_integration_20261031/four_scenes_first_last.png)、[冻结证据索引](reports/full_asset_integration_20261031/FINAL_EVIDENCE.json)。现有5,524原生来源CPU处理及1,567唯一来源cuDex队列全部终结，不要重启。最终语义/尺寸过滤入口为 `runs/semantic_scale_consumer_20261030_v1/final_consumer_v2/BUNDLE.json`：935抓取来源/941变体/26,634 pose，148容器或承托面来源/159变体；这些是候选，不是全库物理成功率。背景4,036来源/3,651独立几何，1,969通过通用地面布景检查，数量要求已满足但真实性仍有差距。

原9对任务独立验收3/9→5/9，包含完整8物成功例；有3改善、1退化，未全局晋级。下放条件修复同3例2/3→3/3，面包横向偏差16.04→6.29mm，原15mm规则不变；6/8物控制例动作状态逐值不变。不能将修后1例拼入旧9例宣称6/9。本轮累计93/128物理全部终态，背景渲染累计137/140全部终态；另8任务仅准备未提交。新API/人类视频/正式配对均0。现场无本轮活动作业；详见 CURRENT_STATE.full_asset_integration_20261030，旧时间点状态继续保留。

下一重点是手掌与容器/已放物碰撞、薄物脱手、尺寸分层采样。935来源中887仍依赖工程尺寸，容量筛选偏向小物；58,117原始记录没有全部转为可抓原生资产。下方“cuDex正在运行”“160容器”等均为历史阶段，不代表当前状态。

2026-10-30 13:36 UTC 后续整合：**现场以 CURRENT_STATE.full_asset_integration_20261030 与实际锁/账本为准**。全部5,524个现有原生来源的CPU阶段已终结；717容器候选已检查，当前可采样160来源/172变体。开源背景4,036来源，去重3,651几何，1,969通过同一地面布景采样/保护区检查；实际视觉覆盖65种，不代表全库视觉已审完。新尺寸cuDex正在原累计3,080额度内处理1,567个唯一源，四卡共享、训练进程不动；不得重复启动旧队列。最新不可变抓姿包由 `runs/metric_assets_20261030_v7/bundles/LATEST.json` 指向，运行结束报告为 `reports/metric_population_20261030/FINAL_STREAM_OBSERVER_V1/COMPLETED.json`。

本阶段实际38次物理尝试已终结：随机多物任务7/12通过，另有局部对照。开爪/撤离候选r1接口失败0/3保留，修正阶段后r2同初态/任务/抓姿3/3通过，原勾倒案例偏差42mm→4.1mm。完整路径端点可达仍不足，正在加入同臂局部连续路径采样和容器容量条件采样。新宽屏参考的比例保持已有本地实测；照明、夹持稳定性、未测资产/容器组合及人类视频仍有缺口。新API0、完整新Real–Sim–Human配对0。**下方同日“21条抓姿待验证”“400背景”“队列全终结”均为较早阶段记录，不是当前状态。**

2026-10-30 全量整合正在运行：用户要求覆盖当前所有资产、真实背景至少1000，并完成尺寸→抓姿→批量物理→完整视频链。现场以CURRENT_STATE.full_asset_integration_20261030为准。背景增量3636源复用旧400（assets/visual_furniture_bulk_20261030_v2）；全部58117来源状态在runs/metric_assets_20261030_v3，首个新尺寸已有21条原生抓姿但尚未机器人任务验收；新物理统一额度128、历史实际141/cap307，新cap435，runs/production_integration_20261030_v1先跑固定3对前缀复用回归。新API尚未登记或提交。不要重复启动已在运行的队列。

2026-10-30 本轮已完成：[16条合格轨迹的成对4×4首末帧图](reports/multitask_20261030_visual_review/paired_index.html)、[四场景真实家具背景对照](reports/visual_furniture_bulk_20261030/comparison.html)、[汇总与使用入口](reports/paired_assets_20261030/README.md)。400来源加工得到385独立几何，292尺寸一致、200地面背景候选，当前四组可抽取148；4作业/24图加载11件家具，主要在对面相机可见。尺寸v2实际生成3个0.9/1/1.1版本，质量/惯量和缓存错配检查完成，新尺寸抓姿与物理资格仍待验证。独立复核已结束，1像素承接面边界差异及分类/亮度限制保留，不宣布全库视觉通过。API、新物理、新人类视频均0；所有作业结束，渲染累计实际80/cap83。现场见CURRENT_STATE.paired_background_expansion_20261030，下方同日记录保留历史时间范围。

2026-10-30 同日后续：在上方落放时全张开夹爪的追加3次物理已完成，独立验收2/3，与贴近放置2/3持平，高于旧窄幅慢开落放1/3；两个成功场景误差约1–5mm。另一个任务在第二次抓取前“预览通过后重规划”失败，第二物未动，不能算掉落。[本轮最新总证据](reports/pragmatic_pipeline_20261030/FINAL_EVIDENCE.json)。当前本实验谱系实际141/登记上限307，本轮9次物理、2次cuDex优化、2次渲染、API0，所有作业终结。旧6次结果及下面较早状态保留；完整配对仍0，不默认晋级。

2026-10-30 本轮已实现尺寸/质量版本预处理、cuDex坏资产跳过、独立背景与上方释放候选：[结果与决定](reports/pragmatic_pipeline_20261030/README.md)。实际4个尺寸版本尚无新抓姿资格；cuDex接续2个作业均结束但0新pose，累计3724/5528、非空2492、pose70062，当前无活动/未知作业；旧v4停止记录保留。背景同轨迹2作业/12图、9场景CPU接线，已加入batch脚本prepare-appearance/run-appearance。低高度释放6次物理对照：相同新版合同下贴近放置2/3、上方落放1/3；修复一例路径，另两例落点/倾倒退化，故不整体替换默认。新物理累计138、上限304，API与新人类视频0。下方“尚未实现/当前3722”等为较早阶段；以CURRENT_STATE.pragmatic_pipeline_update_20261030为准。

2026-10-30 尺度溯源：本批29资产全部沿用target_middle=5.5cm的夹爪归一化；其中4个ABO的商品尺寸/重量、4个GSO原SDF早已存在本地，旧入口未使用。见[源证据](reports/asset_scale_rethink_20261030/RECOVERED_METRIC_EVIDENCE.json)和[修改决策](reports/asset_scale_rethink_20261030/DESIGN_DECISION.json)。下一轮先按来源校核尺寸与用途，再为真正改变尺度的资产建原生几何/抓姿版本；不要全库机械放大或绕过旧缓存绑定。当前仅完成诊断与设计，没有改资产、重启优化或晋级默认。

2026-10-30 最新可视化与复查：[16条轨迹的首末帧4×4集合及视频](reports/multitask_20261030_visual_review/index.html)，覆盖9场景、9种目标结构、20搬运资产来源与7实际使用承接面来源。新发现：16次无完整路径中14次最终卡在初态倾斜物体；S008第二物下放的6条手掌碰撞已用原生几何复现，同姿态“仅容器”通过、“加入已放物”失败。现有packing.py的已放物扫掠未接入本轮工厂，是明确接口缺口。新增7保存状态回放/14视频，API与新物理0；原标签和默认入口未改，未宣称修复后成功。详见CURRENT_STATE.active_multitask_visual_review_20261030。

GitHub 私有仓库：[fufumeichi/ReSimPair](https://github.com/fufumeichi/ReSimPair)。代码、资源和实际私有配置已上传；发布核验见 `validation/github_publication.json`。

先读 [README.md](README.md)、[HARNESS.md](HARNESS.md)、[CURRENT_STATE.json](CURRENT_STATE.json) 和 [迁移验证](validation/RESULTS.md)。

2026-10-30 多物任务试验已完成：[实际结果与根因](reports/multitask_20261030/README.md)、[9 场景／18 个双视角视频](runs/multitask_render_20261030_v2/gallery.html)。10×8 原请求完整保留，75 次执行尝试、5 次预检拒绝、55 条独立物理合格，其中 27 条多物任务；固定 8 对照候选 5 通过、基线 3 通过，存在 1 项退化。每场景合格数为 8、6、0、7、5、3、3、8、7、8；腾空后放置 6/10 有完整物理及原生占位前后证据。入口 `scripts/multitask_pickplace_v3.sh`；工厂 `runs/multitask_factory_20261030_v4/`、执行 `runs/multitask_10x8_20261030_v3/` 已全部终结，禁止重复启动。原规则未放宽，API／新人类视频为 0，未晋级默认。下一步修完整 SE3 传递、native 稳定初态和释放撤离；3 个物体开局倾倒，端点筛选丢失 roll/pitch 已获实证。下方 cuDex“恢复运行”是较早状态：续跑 828 个作业后因新 NaN 输入错误停止，当前非空缓存 2,492、pose 70,062；[只读根因诊断](reports/cudex_nonfinite_20261030/diagnosis.json)指向零体积视觉网格产生非有限质心，没有重启队列。实时状态和后续顺序看 CURRENT_STATE。

2026-10-30 最新接续：[统一批量入口与初始化修复实测](reports/pipeline_integration_20261030/README.md)、[新两场景双视角视频](runs/unified_pickplace_render_20261030_v1/gallery.html)。`scripts/batch_pickplace.sh` 已实际贯通缓存/随机布局/完整路径/CPU执行/审计/双视角重渲染/条件导出：3场景×2任务，6/6物理子项通过，实际抓取3种资产、3容器、12种干扰物；新4个仿真视频、2份条件，API与新人类视频均0。小物体的视觉质量仍是缺口，不能把最低可见性通过当成生成质量。**cuDex原队列已按独立第四轮gate恢复**，当前驱动为 `resimpair_tasks/cudex_seed_repair_campaign_v4.py`，policy/status/counts在 `runs/cudex_bulk_5524_20261028/seed_repair_repro_v4/`。23次诊断占用原总额，剩余2634个首试资产获准四卡并行，末23个因原额度暂缓。条件修复只在原CUDA种子池实际为空时触发；原正常初始化保留。固定相同实际输入，原solver两次44/38、条件版46，证明优化输出自身变化，旧精确历史输出gate继续保留失败。前三失败gate和原32异常不删除；下方“停止”或“尚未GPU验证”均是更早时间点，不能重启旧quarantine入口。现场以CURRENT_STATE与实际PID/锁/新status为准。

2026-10-29 最新接续：[背景质量改进与真实同轨迹对照](reports/scene_quality_20261029/README.md)、[并排图/视频](reports/scene_quality_20261029/comparison.html)。新增用途＋实际RGB选材和米制真实家具候选；3来源×4条件、6完整双视角作业均完成，72张截图/12视频/6份生成条件经原冻结验收代码通过。2/3来源桌面纹理干扰降低，但亮度/色差收益不一致，保留来源仍低对比，不晋级默认、不声称人类质量提升；API与新物理均0。独立agent因额度中断，root接续既定方案并运行冻结验收，不能称已由独立agent完成最终审稿。**cuDex原队列已在32失败停止**：2,871作业终结、1,936非空候选缓存、53,565 pose；下面“仍运行”只是旧时间点。空seed候选已有3真实失败CPU诊断＋5回归，但尚未GPU验证，不得提高阈值或直接重启原队列。优先读`CURRENT_STATE.active_scene_quality_20261029`与[抓姿诊断](reports/cudex_seed_diagnosis_20261029/README.md)。

2026-10-29 最新背景扩充：[实际结果与局限](reports/scene_domain_20261029/README.md)、[完整图/视频对照](reports/scene_domain_20261029/gallery.html)。从旧16材质/9HDRI扩为**221材质/120个带室内标签HDRI**，新增20种程序化三维工作区与有界对面相机；实际20组状态截图、8组双视角完整重渲染和8份相机绑定生成条件通过独立复核。它们复用3条旧物理执行，不计新动作；本轮API与新人类视频均0。20结构实际用了68个新库资源，不能外推全部341资源都已验证视觉质量。独立v4按实际生成参考相机检查bbox/像素并保留辅助裁剪，旧v3的14/20结果保留，两者通过率不能当生成改进收益。候选仍须显式传`assets/appearance_v2/curated/catalog.json`；raw目录仅作历史，旧付费runner尚未接入新`generation_ready`门。当前状态看`active_scene_domain_20261029`，不要重跑已完成28作业；cuDex原5,524队列仍按原规则在后台接续。

2026-10-29 本轮更新：[随机抓放、真实外观与整段规划实测](reports/pickplace_batch_upgrade_20261029/README.md)。扩充快照已自动组合100场景/500目标，涉及200操作来源、74容器和517干扰物；这500项是几何编译结果，不是500条成功轨迹。完整路径v1在相同初态的旧6例从4通过提高到5，4个未调试布局从3提高到4；3mm净空v2出现回程退化，不晋级。新资产入口6请求中4次实际执行通过、2次端点拒绝；后续v5用真实IK接纳布局，30/30端点资格与完整物理结果分开记录。16种纹理/9个HDRI已实际渲染；[四视频同步对照](reports/deferred_render_20261029/comparison.html)来自同一物理执行的两种外观、两个相机，各398帧。两份人类生成条件已通过独立谱系验证，新增人类视频/API调用均0。具体最终执行数与未解问题见报告和CURRENT_STATE，不能从候选实现推断全部默认入口已晋级。

2026-10-29 扩库最新接续入口为 `resimpair_tasks/cudex_quarantine_v4.py` 与 `runs/cudex_bulk_5524_20261028/recovery_policy_quarantine_v4.json`，仍是原5,524清单、5,528 GPU作业上限和原截止时间。仅严格匹配的空Newton输入终结失败可隔离后继续，失败不重试、不计成功；最多32个同类错误或连续3个即停止，未知错误仍停止。它修复批处理阻塞，并未定位求解器根因。实时数量、PID身份和锁看CURRENT_STATE及原目录status/counts；下面v3和更早记录保留历史时间范围，禁止另开重复队列。

2026-10-29 随机抓放与扩库恢复：[无 Agent 批量生成准备程度](reports/pickplace_automation_20261029/README.md)。标准抓放可由程序采样、选臂、规划、验收；宽布局候选已有但入容器小样仍仅1/3，背景主要是颜色。原5,524扩库在986作业终结时因1个空Newton输入异常停止（候选981导出、689非空）；本轮保留该失败并通过版本化隔离恢复剩余4,542个，35测试通过，四卡共享及原额度/规则不变。当前接续入口是 `resimpair_tasks/cudex_quarantine_v3.py` 与 `runs/cudex_bulk_5524_20261028/recovery_policy_quarantine_v3.json`；下方旧入口及数量仅为历史，实时看CURRENT_STATE、status/counts和实际PID，不重复启动。

2026-10-28 已进一步切换四卡共享：GPU0/1/2/3均已实际完成cuDex作业，最多每卡1个优化器、总共4个资产并行。原5,524清单/账本/5528上限/截止时间不变，接续入口改为`resimpair_tasks/cudex_multigpu_v2.py`，资源策略`runs/cudex_bulk_5524_20261028/resource_policy_multigpu_v2.json`。先看CURRENT_STATE.active_cudex_bulk_resume_20261028及队列实时status/counts；下面单卡记录是历史，不要再启动旧单卡监督进程。

2026-10-28 GPU共享已获明确授权并实际启动：原5,524资产队列现由 `resimpair_tasks/cudex_shared_gpu_v1.py` 在GPU2并发1接续，资源策略为 `runs/cudex_bulk_5524_20261028/resource_policy_shared_gpu2_v1.json`。原等待监督进程已完成身份核验后交接，原8资格作业全部完成，原独立gate已通过，现已进入剩余5,520作业；资格候选4资产共119个原生几何合格pose。优先查看同目录status.json/counts.json及CURRENT_STATE.active_cudex_bulk_resume_20261028。下方“等待许可／新增0”属于授权前历史。

2026-10-28 随机化候选已实际接入：[宽范围布局与通用抓放结果](reports/pickplace_randomization_20261028/README.md)。100组原资产/任务生成100个新多物体布局，操作物X跨度49.91cm、117个容器位置、299个桌面目标；100/100几何编译成功。预先固定6次CPU物理：桌面3/3、入容器1/3通过物理子项；两例搬运失败原因已留档，完整视频配对0。候选不晋级默认；cuDex新增作业仍0、原队列继续等空闲GPU。先查CURRENT_STATE.active_spatial_randomization_20261028。

2026-10-28 用户已将优先级明确为资产扩充、随机化和通用抓放规划：[实际随机化审计与扩库接续](reports/pickplace_randomization_20261028/README.md)。原100场景全用forward模板，操作物X跨度1.6cm、容器3固定位置；干扰物已用165来源。5,524资产几何准备已完成，新cuDex作业0，队列因既有不占用GPU边界等待空闲；GPU2共享资源选项已询问，尚未授权生效。先读CURRENT_STATE.json.user_priority_20261028_asset_randomization，不以新增审核模块替代该主线。

2026-10-28 总体整理：[批量生产准备程度与非资源缺口](reports/pipeline_readiness_20261028/README.md)。已核对原500请求343通过、39/100场景五任务全通过；127保留抓姿来源＋1恢复来源不等于5,524个原生几何就绪资产或同量物理成功。抓放之外8机制小批已有70条合格仿真。当前主要缺口是整段路径前置、角色/尺度语义、镜头/事件条件、独立审核和同一批次的完整配对产率。此次只有审计与整理，无新API、物理或优化作业，已有队列和实验状态不变。

2026-10-28 首尾条件对照已完成：[实际视频、质量与计费](reports/endpoint_conditioning_20261028/README.md)，[并排播放](reports/endpoint_conditioning_20261028/comparison.html)。同一苹果入碗执行新生成 A 首图参考＋仿真、B 首尾图参考＋仿真、C 专用首尾帧三条8秒视频；单条约4分半，视频名义扣费分别 $3.991680 / $3.991680 / $3.345462，含三次辅助审核合计 $11.855870，日志与额度扣减一致。独立复查均支持九维，但 A 本就满足首尾状态，所以未证明新增尾图的收益，不扩第二布局、不晋级默认。首尾图复用，新增生图及物理执行均0。VLM 的分区混淆、左右手误报保留并用实际绑定/连续帧裁决。真实首帧增物失败的单变量输入已准备但未运行；投影中心越界不能当作整个物体不可见。先读 CURRENT_STATE.json.active_endpoint_conditioning_20261028，禁止重交已完成调用。下面早期汇报和 cuDex 队列状态仍保留其时间范围。

2026-10-28 最新汇报：[11页简明黑体版：扩量能力、阶段耗时与费用](reports/pipeline_capacity_cost_20261028_v3/README.md)已完成，PPT含4段原始视频，详细token与预算单列Excel，旧48页版保留。已核实最近3例只选桌面移位/分组；真实视频请求使用参考图＋参考视频，没有专门首帧/尾帧条件，并存在视野外物体文字冲突。新增物体失败已图示，成因消融和生成修复尚未验证；后续用同一入碗任务做条件方式对照。报告新增API、GPU、物理执行均为0，不替换下方全量抓姿队列现场。

2026-10-28 全量接续：[cuDex中断修复与5,524资产队列](docs/CUDEX_BULK_RESUME_20261028.md)。CPU原生加工已全部结束，5,524个规范资产就绪（含别名覆盖5,550来源）；旧八作业队列实际执行1次后因ESDF/碰撞代价共用缓冲错误中断，七个未提交。v3隔离缓冲并保存完整求解结果，旧结果恢复出1个新来源7个原生几何合格pose，单独可迁移包在 `assets/precomputed_grasps_recovery_20261028/`。全量队列、资源等待和实际新作业数请先查 `CURRENT_STATE.json.active_cudex_bulk_resume_20261028` 及其队列状态。旧127缓存与恢复1的来源并集128，不是物理成功数；原生就绪5,524也不是pose成功数。全量协议最多新增5,528 GPU作业、API 0，资格检查通过后自动继续，禁止重复开新目录。

2026-10-28 最新后续：[cuDex原生匹配、CPU物理验证、全量资产与新人类样例](docs/SIMULATOR_ALIGNMENT_20261028.md)。11,132来源/11,082几何作业的后台CPU队列持续运行，先查 `CURRENT_STATE.json.active_alignment_20261028`、PID和最新状态，禁止重复启动。路径回归30/49；固定6例的26例版本已做真实CPU无渲染执行，同新v2规则物理分项0/6→6/6，RGB缺失故完整验收0/6。127仍是保留的抓姿缓存数，新增native-ready不能当新缓存。cuDex约束代码已修候选、8作业等空闲CUDA；当前无证据宣布全库新成功率。3新人类视频已完成，名义API扣减$12.723492，首例对象对应失败、其它例局部支持；旧503unknown保留。本轮详细黑体规划PPT在 `reports/data_scale_plan_20261028_v2/`，旧PPT不覆盖。

2026-10-28 最新完成：[资产容量与接触微调结果](validation/asset_capacity_20261028/RESULTS.md)。原500请求全部终结，343通过，105来源稳定抬升、102来源完整任务通过、42明确目标容器通过；旧88/86是312请求冻结时的覆盖。对49个空缓存保持原规则检查完整256候选只恢复1来源；原生指接触有限平移后恢复17来源、260候选，已打包127来源＋78容器到 `assets/precomputed_grasps_contact_v1/`。后续14请求、12次实际执行、6通过，新增3个原空缓存来源完整通过；原500与小样覆盖并集108抬升/105完整通过，不是统一成功率或任务类别数。两组物理的规则均与原500一致，旧库/失败保留，候选未晋级默认。共享物理额度560/560、历史承接119，全部预约终结；GPU pose作业仍141/256，新增API0。先查 `CURRENT_STATE.json`，不要再次启动已完成的500或两个小样。下一步优先原生碰撞转换和执行前整条抓放路径筛选；现有物理额度已用完。

2026-10-28 较早诊断快照：[资产容量与 cuDex 接口诊断](docs/ASSET_CAPACITY_DIAGNOSIS_20261028.md)确认旧 PPT 的 88 个是 312 请求的覆盖截断；同一方法到465请求已有103来源稳定抬升、100来源完整任务通过。新256来源中115在原生碰撞转换被挡、49优化后缓存为空、2优化失败、90进入缓存；目标约100缓存与固定抽256是上一轮范围，不是库上限。24,000已处理网格中有11,461离线候选、11,132未进入当前/既有选择；不能称为同量动态成功。CPU同规则检查256候选在一个旧空缓存来源的rank189找到合格候选；另有TCP截面测宽与真实手指接触定义不一致的反例。cuDex原生TCP和双指对称性未发现错误，但upstream success与本地物理证据必须分开。原500调度在465后因GPU占用退出，确认无在途及资源空闲后同根恢复。现已完成，原规则和旧PPT快照未改。

2026-10-27 最新：[固定 500 任务候选](docs/DIVERSE500_CANDIDATE.md)已开始实际批量运行，[阶段结果](validation/batch500_20261027/RESULTS.md)保留所有资格与失败。110 个原生缓存资产、78 个容器组成可迁移运行包；100 场景×5 个目标使用109个操作来源，仍是抓放机制，不能称为500种机制。新20请求资格从13/20到18/20，旧13成功无退化；4条实际用了SE(3)搬运补救，另1条此次原分支成功。满足原门槛后正执行剩余480，实时以 `runs/batch500_20261027/production_transport_v2/progress.json`、实际PID与锁为准；出现 `completion.json` 才是全批完成。新来源已暴露抬升、夹持、容器下放和目标TCP工作区问题，当前固定方法不边跑边改。共享物理额度560，单GPU2，默认控制未晋级。

本轮人类小样实际完成首图1、视频1；抽帧支持单人及右手搬运，细物体身份/遮挡接触仍unknown。第二视频503无ID，保持未知且没有重交，剩余付费小样停止扩量。实际观察名义消耗$4.2502，首视频提交至下载313.10秒；人民币实付未知。累计调用image31/video25/review55，包含本轮image1/video2/review3和失败提交。正式Real–Sim–Human三元组0。详见 `runs/batch500_human_pilot_20261027/repairs/schema_wire_v2/README.md`。以下早期同日结果与预算仅代表各自历史阶段，不替代最新现场。

2026-10-27 当前入口：[指定目标通用抓放](docs/REQUESTED_PICKPLACE.md)与[实际结果](validation/requested_pickplace_20261027/RESULTS.md)。使用已有 cuDex 缓存，由同一脚本接收物体 ID、源位置、目标位置／容器，自动检查尺寸、内腔与可达性。8 来源、17 任务：原配置 13/17；2 mm 接触配置 15/17，修复三个支架放置但新增灯具接近失败，故仅保留可选候选。默认仍是 legacy_v1＋native_default。原始请求目标独立验收已接到 R2H 导出，实际导出 2 份条件，API 0、人类视频 0。新执行40次，当前共享75/84、谱系累计119/128；所有执行已结束，剩余9次不用于重复旧实验。下一步复用现有 Cartesian 接近／抬升求解并用新冻结布局验证；goose 抬升仅 CPU 回放通过，未物理验证。

同日较早阶段：[64 资产三方案对照结果](validation/grasp_comparison_20261027/RESULTS.md)和[预计算抓姿运行](docs/PRECOMPUTED_GRASP_RUNTIME.md)。实际 GraspGenX / cuDex / 混合候选覆盖为 24/64、44/64、48/64，完整抓放为 7/7、12/13、10/13，所有预检失败另计；按同样预登记 26 来源计为 7/26、12/26、10/26。准确来源划分为 2 开发 + 61 未见 + 1 历史资产回归。三方案原始 pose 已存入 `assets/precomputed_grasp_research_v1/`，17 个原生资产的直接运行包在 `assets/precomputed_grasps_v1/`。`cache_runtime` 只读缓存、不导入优化器；独立克隆、隐藏原工程且断网实跑 2/2 通过并导出 2 份 R2H 条件。代码与缓存已按用户要求发布私有 main（实现提交 `dc67eb7`）；发布核验见 CURRENT_STATE，方法验证范围不因进入 main 扩大。

上述较早阶段 API 0，新增人类视频和正式配对 0。GPU 资产方法作业 195/198；新增物理 35/84，承接此抓取实验谱系 44，累计 79/128，未知提交和活动作业均为 0。预算余量不代表应重复已结束试验。后续优先处理 9 个碰撞加工失败、4 个实际接触/路径失败，再扩未调试布局；复用进程与采集接口以减少约 22.7 秒成功轨迹中的渲染、状态读回和启动开销。原验收规则和三方法参数均有冻结哈希。

同日较早的历史阶段：[实际 cuDex 原生抓放与收集对照](validation/cudex_physical_20261026/RESULTS.md)。实际 CUDA 优化候选在相同八来源／初态下完整抓放 8/8，几何基线 7/8；两组均观察到 8/8 抓持，差异来自抬升路径走廊。原优化器严格 `success` 标记均为零，与打印的分数计数不同，不能混报。新增双物体同容器收集 2/4；3 个真实抬升失败通过 CPU 笛卡尔重规划和原守卫，修复后的物理执行尚未验证。当时 10 份 R2H 条件已导出，API 0，新人类视频和正式配对 0；新增物理 20/20、累计 44/44，GPU 优化 10/10，包含 2 次接入失败。上述历史额度已由最新谱系承接，不能清零。该轮使用 1,000 种子，与最新 256 种子协议分别统计。[接入说明](docs/CUDEX_NATIVE_PHYSICAL.md)。

这是当前 Real→Sim→合成人类参考管线的独立副本。主入口是 `scripts/run.sh`，主线模块在 `corl2026_workshop/experiments/autonomous_pipeline_v4/`。`provenance/original_project/` 是迁移前的历史状态，不是这个仓库的当前路径配置。

2026-10-26 的资产与通用抓取候选在 `resimpair_grasp/`，通过 `scripts/run.sh grasp` 运行。先读 [使用说明](docs/GENERIC_GRASP_PIPELINE.md) 和 [实际对照](validation/generic_grasp_20261026/RESULTS.md)。加工资产与缓存已放在仓库内；新实验不需要访问原始 `/mnt/sdf` 工程或外部大资产目录。候选方法的验证范围与原默认方法分别记录。

同日后续的[资产扩量与任务工厂](docs/ASSET_LIBRARY_AND_TASK_FACTORY.md)在 `resimpair_assets/`、`resimpair_tasks/`。原始库有 58,117 条记录，旧 28 个只是加工子集；新候选目录含 258 个来源 ID、256 个静态干扰物资格。实际批处理、纹理修复、多任务和新资产抓取结果见 [本轮记录](validation/asset_scaleup_20261026/RESULTS.md)。不要把静态资格或元数据名称当成抓取、功能或语义确认。扩充尚未加工的资产仍需原始源库，使用随仓库提供的加工资产则不需要。

同日最新的[批量桌面 pose 库](docs/BULK_TABLETOP_POSES.md)在 `resimpair_tabletop/`：12,000 个网格已离线处理，5,767 个资产含可行抓取候选，共 11,304 个支撑姿态、225,870 个抓取 pose。该层不需要逐个动力学仿真。碰撞模型按任务需要转换，物理效果只做预登记小样本检验；不能把这些离线数字与旧的 19 个精加工候选直接混为同一资格。最新结果与抽样证据见 [批处理记录](validation/tabletop_bulk_20261026/RESULTS.md)。未抽样资产保持 `unmeasured`，默认主线没有替换。

以上 12,000 个是第一批历史结果。随后完成的[通用抓放与容器候选](docs/GENERAL_PICKPLACE.md)在 `resimpair_pickplace/`，入口为 `scripts/run.sh pickplace`：累计离线处理 24,000 个网格，11,461 个资产有抓取候选；512 个容器网格筛得 100 个内腔几何候选。已实现宽布局、联合源/目标选臂、真实容器和同场景关系任务组合，并实测可选 GraspGenX。分组物理结果、真实失败及资源消耗见 [新记录](validation/general_pickplace_20261026/RESULTS.md)。几何数量不代表原生动力学成功，候选没有替换默认主线。仓库内 `assets/general_pickplace_demo_v1/` 可直接用于小样运行。

启动前核对当前进程、GPU 占用、`runs/` 和 `runtime/campaign_registry/`，不要重复运行已保留的尝试。运行 `scripts/run.sh doctor --gpu <空闲设备>` 检查依赖；它不调用收费 API。

当前可批量运行有限机制模板、不同布局和同场景多任务，并同步保存动作、状态及双视角。真实视频抽取和外部规划协议入口保留；任意新视频到任意可执行机制仍需要适配。没有真实来源绑定的输出必须保持 `synthetic_only`。

API 配置在 `private/.env.local`；按所有者要求保留并纳入 Git，本仓库只发布为 private。不要在输出、提示词或文档中打印其内容。

新会话提示词：

> 读取 START_HERE.md、HARNESS.md、CURRENT_STATE.json 和 validation/RESULTS.md，核对进程、锁与真实产物，从当前未完成事项继续。使用这个仓库的相对路径入口，不访问或改写原工程。保留已有有效成果、失败、预算与未知状态，不重复提交模型请求，不启动策略训练。区分运行可用、物理合格、视觉采样支持与正式配对成立。

2026-10-26 后续的[夹持、支撑与释放调查](docs/GRASP_RELEASE_STABILITY.md)已完成候选实现和小样验证：实体手指闭合检查、朝向支撑预检、实际负载更新、分阶段释放，以及可选快速 CPU 碰撞转换。详见[结果](validation/grasp_release_20261026/RESULTS.md)。支架夹持倾斜改善，但原鱼缸底面仍不合格；保留验证 2/2，不能外推全库。共享物理额度现为 24/24，旧验收、失败和默认主线保留。

2026-10-26 最新的 [cuDex-graspvla 调查与批量瓶颈分析](docs/CUDEX_GRASP_QUALITY_AND_BATCHING.md) 已完成：精确核对 Franka 延长指／DFC 配置；8 个来源做离线接触搜索与静力代理对照，6 个产生候选。新代理排除了一个既有物理成功来源，故仅保留研究候选，未替换默认。新增只读批次耗时与结果恢复工具；本轮物理、GPU、收费 API 均为 0，详细证据见 [结果](validation/cudex_contact_quality_20261026/RESULTS.md)。
