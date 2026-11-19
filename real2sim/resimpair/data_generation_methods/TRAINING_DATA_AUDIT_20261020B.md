# Real → Sim → 合成人类参考：训练数据接线审计

日期：2026-10-20。范围：只读审查当前源码、真实来源规格、6条原生输入和既有苹果修复视频；运行CPU读取/数值探针，没有调用生成API、训练模型、启动仿真或修改生产数据。本文件是接口审计，不是数据发布批准，也不评价同一工作会话中随后新增的样本。

**结论：合成人类视频可以作为独立时钟的视觉任务参考，机器人原生观测/动作可以继续作BC监督；现有生产reader尚没有合成来源的完整准入接线。** 已实测修复视频的32帧读取正确，但“能解码”“任务语义候选通过”“来源绑定”“实际合法配对池”“获准训练”是五个不同状态。目前不能从其中一个推导另一个。

## 1. 已核实的数据及边界

真实来源规格：

`/mnt/sdf/datasets/hypericl_human_robot_task_library_20261019_v1/generation/prepared_v2/human_grounded_task_spec.json`

本次读取SHA256：`5a861cd4d3b012312564c071d56f49a3b8936235f0b91c8a4b32f187382ce05d`。

| 任务 | 真实窗口 | 本轮原生输入 | 已知限制 |
|---|---|---|---|
| 苹果：occupied bowl → unique empty bowl | 3段train，246/281/287帧，30Hz | 2资产、2活动臂，302/327帧，20Hz | 目标是角色关系，不能把所有布局写成固定右→左 |
| 闭合书本：桌面 → 开盒 | 3段train、2段validation，原视频中的完整原子窗口 | 2资产、2活动臂，411/371帧，20Hz | 不是整个declutter；第二资产的原生`goal_key`有`_new_assets_v1`后缀 |
| 右柜门：closed → open and released | 2段train、1段validation；裁剪后72/72/54帧，30Hz | 2布局、同一柜门资产、左活动臂，499/504帧，20Hz | 不是取出物体、关门或完整insert/remove；不能声称覆盖两臂/两机制 |

6条输入来自 `data_generation_methods/pilot_inputs/input_manifest.json`，均是既有24条pilot中的train机器人轨迹，不是本次新造机器人轨迹。所有源row均显式保留 `target_start_offset=0`。本次分别从真实NPZ读取中段16步，得到state18、native_commands16、EEF14，224个有效action元素。没有把人手姿态当机器人动作。

既有修复视频：

`/mnt/sdg/datasets/real_sim_human_pipeline/repair_study_20261020_v1/role_grounded.mp4`

本次全解码为193帧、24Hz、960×960，PTS为0至8.0秒，`time_base=1/12288`，相对`i/24`最大误差为0。片长按末帧加一个周期为8.041667秒。调用生产 `hypericlzq.media.frames_at()` 读取32帧、224×224，与独立顺序解码后相同缩放的RGB逐像素相等。最终模型张量应为`[32,3,224,224]`。

实际32帧索引为：

```text
0,6,12,19,25,31,37,43,50,56,62,68,74,81,87,93,
99,105,111,118,124,130,136,142,149,155,161,167,173,180,186,192
```

此检查只说明媒体路径/时钟/取帧可用，没有证明全部中间帧的任务质量。`repair_quality_observation.json`仍明确为候选语义通过、`quality_release_approved=false`、`training_admitted=false`。旧的 `pilot_20261020/task_contracts/apple_bowl_transfer_ego_2174.binding.json`还是`prepared_not_evaluated`；该binding本身不能作为修复视频已经经过通用契约门控的证据。

## 2. 真正的读取路径

目前追加任务的实际H2R入口是：

```text
hypericl_data_factory.append_task_snapshots_typed_v2.MultiTaskPolicyRouteDataset
  → next_task_book_reader.BookPolicyRouteDataset
  → training_routes.PolicyRouteDataset
  → training_routes.ReferenceMediaDataset
  → hypericlzq.data.CuratedDataset._item()
      reference RGB：自己的start_seconds、fps、length
      query RGB/state：机器人自己的t
      action：hypericlzq.eef.eef_actions(query, robot_arrays, t, horizon)
```

关键函数：

- `PolicyGen/examples/baselines/hypericl_data_factory/training_routes.py:30`：`candidates()`。
- `PolicyGen/examples/baselines/hypericl_data_factory/training_routes.py:50`：`PolicyRouteDataset`构造候选池并强制传入所选参考。
- `PolicyGen/examples/baselines/hypericlzq/data.py:180`：`CuratedDataset._item()`组装模型输入。
- `PolicyGen/examples/baselines/hypericlzq/eef.py:74`：动作坐标和时间偏移。

该路径按query的action space编译参考程序，不要求人类参考与query逐帧对齐。因此24Hz/8秒人类参考配20Hz/15秒机器人query本身没有矛盾。参考只负责表达完整任务；监督始终取query自己的数组。把生成视频强行拉成原生robot帧数，或继承robot的腕部/接触标签，反而会制造错误监督。

`human_to_robot_pipeline.verify_batch()`的输出是CPU验收与真实源候选绑定；`r2h_pipeline.cli.export_pairs()`的输出是合成人类候选配对；二者都不是上述reader已经使用的训练manifest。

## 3. 已定位、可复现的接线缺口

### A. 来源契约和显式配对边没有进入实际参考池

位置：`training_routes.candidates()`、`PolicyRouteDataset.__init__()`。

当前H2R只筛同`goal_key`、不同`group`、人类`domain`；同split由`CuratedDataset`先行过滤。它不读取`paired_candidates.jsonl`、`allowed_query_ids`、真实来源窗口绑定或`task_evidence`结果。

这不仅是潜在推断。既有文件：

`/mnt/sdf/datasets/hypericl_human_robot_task_library_20261019_v1/training/h2r_goal_aliases_plus_pilot24_v1/source_spec_vs_actual_pool_difference.json`

记录了24条pilot的source-spec关联64边，实际训练reader为80边，交集52、actual-only 28、spec-only 12。此前这些额外参考有单独复核，**这不是既有80边错误的证明**，但确证了“写了来源绑定”并不限制reader池。

另一个具体影响是新书资产：源规格的canonical goal是`factory:closed_book_table_to_open_box`，而原生row为`factory:closed_book_table_to_open_box_new_assets_v1`。若合成参考机械抄canonical goal，它不会配到该新资产query；若把两个goal直接合并，又会扩大其他池。

建议：新增单独的合成参考route/adapter，读取hash绑定的显式边表，最终池取“基础split/goal约束 ∩ 被审核允许的query-reference边”；需要跨goal的资产抽象关系应作为显式关系类型记录。导出后比较**所有实际池的ID集合**与预期边集合，不只检查随机抽到一对正确。

### B. 生产reader没有fail-closed的合成准入，且新source会直接报错

位置：

- `hypericlzq/data.py:13`：`source_id()`枚举只支持现有来源，没有`synthetic_human`。
- `append_task_snapshots_typed_v2.py:69`：新row参考门控只排除显式`reference_pool_eligible=False`；该字段缺失默认保留。
- `training_routes.py:30`：不读取`quality_release_approved`、`training_admitted`或`reference_training_approved`。

只读内存探针实际结果：

```text
source_id({"source":"synthetic_human"})
→ KeyError: 'synthetic_human'

candidates(query, [same_goal/different_group/human reference], "human_to_robot")
→ [0]
```

第二个fixture同时设置了`reference_training_approved=false`、`quality_release_approved=false`、`training_admitted=false`，且`allowed_query_ids`不含query；这些字段均没有阻止其进入基础池。它没有被写入生产manifest，也没有启动训练。

建议：给合成来源分配稳定独立source ID，视觉`domain`仍可使用human，避免伪装为egodex/imitator_human。新增参考必须由独立导出器验证媒体hash、任务审核、准入证据与边表；缺一即排除。隔离CPU读取审查可以读取候选，但它的“允许审查”标志不得等同正式训练批准。`SourceGoalSampler`也需要明确配置合成bucket；默认比例没有该来源，不能以新增row数量推断其实际曝光量。

### C. 派生来源没有进入跨split检查

位置：`append_task_snapshots_typed_v2.py:122`的`check_split_integrity()`。

该检查覆盖当前row的source/group/session、已声明媒体hash、locator、arrays和receipt路径。未遍历真实源窗口→仿真rollout→锚图→多个生成重试的派生链。

内存fixture：两个合成row分别是train/validation，拥有不同group，但相同`lineage_root_ids=["real_human_2174","robot_A"]`。现有函数返回`passed=true`、`cross_split_conflicts=0`。这证明现有函数忽略该字段；本次未发现或声称生产已经发生此类泄漏。

建议：导出前建立带类型的根身份，至少包括真实源session/原视频hash与窗口、源robot rollout fingerprint/初态group、锚图hash、生成family/retry身份。按共享派生根的连通分量固定split。若“不同真实视频但共享同一robot初态”要作为独立切分轴，也应显式声明。不能把每个生成任务ID当作新的独立来源。

合成参考的`group`必须至少继承其源robot group，阻止同一轨迹视觉改写后作为该query的“独立”参考。如果实验有意使用同rollout参考，应单独路由、单独报告，不能混入cross-group结果。

### D. 导出时丢失offset会静默错一帧

位置：`hypericlzq/eef.py:74`。

`eef_actions()`对`source!="imitator_sim"`默认`target_start_offset=1`；当前6条`simulation_expansion`row均显式设置为0。新导出器若只拷常见字段漏掉它，动作起点就从`t`变为`t+1`，不报维度错误。

本次逐条中段16步比较“保留offset=0”和“删除字段”，全部出现非零标签差；最大绝对元素差分别为0.007753、0.002448、0.008968、0.007042、0.057217、0.057229。因为不同元素单位不同，这些数值只证明标签改变，不作为统一物理误差指标。

建议：机器人query整行保持不变，尤其保留`source`、`action_space`、`label_kind`、`target_start_offset`、`fps`、arrays、normalizers和mask。新adapter要求显式offset，禁止回落默认值。动作仍是`R(t)^T(p_target-p_t)`与`Log(R(t)^T R_target)`、每臂open信号；它们是query局部TCP位姿偏移，既不是相机像素，也不是速度，不能乘/除生成视频速度。

### E. 候选媒体缓存身份不含内容hash

位置：`hypericlzq/reference_cache.py:24`的`row_key()`。

key仅取`id/video/start_seconds/fps/length`，不含`source_video_sha256`或实际被`ReferenceMediaDataset`替换使用的`reference_video`。内存探针改变上述hash和reference locator，key仍为同一值。由此，原地修复同路径视频或替换native参考流可能复用旧tokens；当前不可变文件工作方式减轻了此风险，但缓存接口没有强制保证。

建议：缓存key使用最终实际媒体locator、验证后的内容hash、窗口与时钟、采样规则版本；合成每次重试使用新ID/路径，禁止覆盖失败样本。已有缓存可以维持旧版本，新合成数据使用新版本目录。

## 4. 时间、帧、坐标与观测审核的具体边界

1. **窗口不能丢失偏移。** 书本train窗为`[190,261)`、`[175,247)`、`[105,171)`，均30Hz。引用完整原视频时，分别用`start_seconds=190/30,175/30,105/30`和`length=71,72,66`；若明确裁成新文件，则新媒体时钟从0开始，同时保存原窗口映射。柜门输入已是`atomic_native.mp4`，不能再次叠加原长视频偏移。
2. **人的时钟和机器人的时钟独立。** 合成193帧/24Hz与robot 302–504帧/20Hz分开记录；`action_dt`必须来自robot的0.05秒。只给语义配对时，`frame_alignment=false`，不要写“同帧ground truth”。若以后测motion，事件/时间对应必须另有可复算依据。
3. **生成图像不继承仿真坐标真值。** `task_evidence.evaluate_task_evidence()`正确禁止用`sim_measured`认证human视频。其`paired_2d_tracks_v1`只检查已有共享像素坐标、显式时间配对下的残差/覆盖；不自动配准不同相机，不验证tracker身份正确，不证明接触、3D或human wrist GT。
4. **审查采样与训练采样不同。** 当前32帧训练索引不含坏样的第1–3帧异常窗口。8帧/密集边界审核可以拒收坏样，但训练reader不会自动使用审核帧。发布前必须直接看实际32×224输入是否仍保留抓取、放下、收手及正确目标，而不是只看960像素审核图。
5. **只有全片被解码还不够。** `task_evidence`是bound报告门控，不自己解码媒体，其duration来自证据；生成导出器应将full-decode得到的PTS、frame count、duration与evidence绑定并比较。VFR不能仅拿平均fps套`idx/fps`；可拒绝不符合CFR的候选，或实现独立PTS取帧。现有本例实测为CFR。
6. **参考侧不要携带机器人动作。** 合成row应为`action_space=null, arrays=null`，腕部/接触/力GT为false。两条参考如果只是同一源rollout的不同生成重试，不应因不同文件名就令`pair_valid=true`，也不该按独立正样本累加。

## 5. 可执行的最小接线方案

保留当前真实→仿真采集和验证器；在它们与生产reader之间增加隔离的合成候选adapter，不修改旧生产清单或prepared pointer。

```text
真实任务spec + 精确人类window
  → robot receipt/arrays/media + 真实源binding
  → 合成视频 + generation lineage + 完整解码时钟
  → task evidence decision + 独立发布审核
  → synthetic reference rows + 显式query/reference边 + split/lineage审计
  → CPU actual-reader probe
  → 新版本发布manifest（当前仍不训练）
```

候选参考row至少记录：

```json
{
  "source": "synthetic_human",
  "domain": "human",
  "action_space": null,
  "arrays": null,
  "fps": 24.0,
  "length": 193,
  "start_seconds": 0.0,
  "reference_pool_eligible": false,
  "quality_release_approved": false,
  "training_admitted": false
}
```

还必须有唯一id、原始split、源robot group、scene、精确goal或经审核的关系类型、不可变video locator/hash，以及真实窗口/robot receipt/锚图/生成尝试/审核证据的hash绑定。上例只是候选字段示意，不能直接加入训练。

复用已有 `hypericl_task_library.prepare_training_pilot.human_expected()`和`probe()`的独立核验思路，完成以下读取验收即可判断“接口接通”，无需训练：

- 枚举预期与实际全部边，验证集合相等；明确单独统计不能跨goal配到的新书资产。
- 在每个robot query的开始、中段、最后有效动作附近强制读取被允许的合成参考；确保`reference_episode_id`确实是新视频。
- 独立顺序解码合成媒体，逐像素对比实际32帧；反归一化action/state与原生NPZ公式比较，检查末尾horizon mask。
- 构造拒绝样例：未发布参考、错goal/错split、同robot派生根、错误媒体hash、丢offset；要求在取batch前失败。
- 再确认旧row/normalizers/既有池完全未变。审查通过的临时adapter也不能自动宣布正式质量发布。

去重和数据规模应同时报告：真实人类独立session/窗口数、robot物理rollout数、唯一初态group数、合成文件数、生成attempt数、候选语义通过数、正式参考数及实际训练边数。当前8段train人类源派生192个计划robot任务或更多合成文件，不会变成192段独立真实人类演示。

## 6. 本次证据强度

已实测：6条NPZ中段动作读取、offset遗漏敏感性、修复视频全解码与真实reader32帧RGB一致性；source枚举、准入筛选、lineage split、cache key四个最小内存探针。

已阅读但未在本次重跑：既有24条pilot的完整H2R actual-reader报告、scale192批验收流程、FK/接触检查与所有真实人类视频逐帧核验。本次没有把这些历史结果说成本次新增的全面验收。

尚未完成：新的`synthetic_human`生产batch贯通、通用生成候选发布、跨源lineage隔离验证、多任务合成质量统计和训练收益评估。当前最有价值的下一步是完成隔离adapter及上述实际输入检查，再决定哪些新样本具备训练资格。

## 7. 同日后续已实现的隔离接线

本审计完成后，按任务追加实现以下新文件，未改旧生产reader：

- `data_generation_methods/r2h_pipeline/training_pairs.py`：`validate_pair()`、`validate_collection()`、`lookup_pair()`、`reference_cache_key()`、`read_pair_inputs()`、`inspect_pair_inputs()`。
- `data_generation_methods/tests/test_training_pairs.py`：临时artifact测试准入、显式边、lineage跨split、cache内容身份、独立时钟、原生offset、窗口越界、真实CPU张量和末端padding。
- `data_generation_methods/scripts/compile_batch_training_pairs.py`：只编译已经落盘的本批视频；每次产出不可变内容寻址快照，缺失视频列为pending。

新validator要求原robot row与hash绑定的原manifest逐字段相等，原生offset显式为0，artifact路径/hash匹配，合成参考不能带action arrays/wrist3D标签。训练模式必须同时通过pair/ref两层quality release与training admission；inspection即使面对已有批准也返回`cannot_train=true`。cross-demo检查继承源robot group，禁止源query同组；不同书资产的goal关系只能显式声明并附证据，未批准的关系仅允许inspection。

固定读取函数返回真实CPU NumPy输入，robot action来自其原始数组及原normalizers。编译器再调用**未改动的**`hypericlzq.media.frames_at()`与`eef.eef_actions()`核对RGB、动作、state和mask；不调用完整旧`CuratedDataset`，不隐藏其source枚举尚未升级的事实。

已先实测本批白碗合成参考→黑碗机器人query，后者t=0/151/301三次读取均成功：reference `[32,3,224,224]`、obs `[3,224,224]`、action `[16,20]`（14有效维）、state `[20]`（18有效维）、robot dt=0.05、offset=0；末帧仅1步14个action元素有效。RGB与生产取帧函数逐像素相等，动作/state在`atol=3e-6, rtol=2e-6`内相等。

快照根目录为：

`/mnt/sdg/datasets/real_sim_human_pipeline/batch_20261020b/training_pair_inspection_snapshots/`

每个快照含`pair_manifest.json`、`fixed_sample_cpu_inspection.json`、`summary.json`，汇报本次实际发现的视频、pending、拒绝原因和固定样本数。复跑命令：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
PolicyGen/.venv/bin/python -B data_generation_methods/scripts/compile_batch_training_pairs.py \
  --input-manifest /mnt/sdg/datasets/real_sim_human_pipeline/batch_20261020b/input_manifest.json
```

这些接口修复使候选数据能够被隔离、显式、可核验地读取；它们没有授予视频正式质量资格，也没有启动训练。

## 8. 四视频完成后的实际候选配对验收

最终本轮读取快照：

`/mnt/sdg/datasets/real_sim_human_pipeline/batch_20261020b/training_pair_inspection_snapshots/ac81e404ed146b5638ca/`

- `pair_manifest.json` SHA256：`73c7e845e5dfe7fa9c0cc0282649af953c071465ff98126ad27e2eda963b550f`
- `fixed_sample_cpu_inspection.json` SHA256：`52a0e15db5f151a70fd35704d6ee48a009aa334ae88c826aa7e6f1dd53cf1b73`
- `summary.json`：4个已存在视频、0 pending、4条显式候选pair、12个固定样本读取、0读取/数值校验拒绝，`cannot_train=true`。
- `source_snapshot/`保存本次编译器、隔离adapter和实际参与对照的原生产media/eef源码，避免后来编辑源码破坏历史复现。

四条参考均独立全解码为24Hz、193帧、960×960、8.041667秒。所有query保持原20Hz、`target_start_offset=0`；没有根据生成时长重采样机器人动作。

| 合成参考的源机器人 | 显式配对query | 实际query帧 | query长度 | 归一化action最大绝对差 | 归一化state最大绝对差 |
|---|---|---|---|---|---|
| 黑碗891910000 | 白碗891911000 | 0 / 163 / 326 | 327 | 4.89e-8 | 9.26e-8 |
| 原书891912000 | 新书资产891913000 | 0 / 185 / 370 | 371 | 1.14e-7 | 4.51e-8 |
| 柜门891914000 | 柜门另一布局891914001 | 0 / 252 / 503 | 504 | 8.86e-8 | 4.33e-7 |
| 白碗891911000 | 黑碗891910000 | 0 / 151 / 301 | 302 | 9.77e-8 | 7.84e-8 |

上述差值是隔离固定读取器与**未改动的生产EEF函数**的归一化张量比较，反映浮点计算精度，不是生成视频与真实人手的物理误差。全部12次reference/query RGB与生产媒体函数逐像素一致，action/state/mask匹配；各query起始与中段有224个有效action元素，最后一帧只有14个，尾部15步保持mask=false。实际shape一致为reference `[32,3,224,224]`、obs `[3,224,224]`、action `[16,20]`、state `[20]`，14维机器人action和18维state之外都是padding。

三个exact-goal配对之外，书本对明确保存了`explicit_task_only_equivalence`：reference为generic book goal，query保留`_new_assets_v1`。该关系的`training_approved=false`，没有改写源goal，也没有扩展其他query池。

**读取通过4/4，不能写成自动质量合格4/4。** 编译完成时，本批审核文件实际状态如下：

| 参考 | 自动审查实际结果 | 尚缺内容 |
|---|---|---|
| 新黑碗 | `task_review/gate_decision.json: needs_review` | source bowl、target bowl移动和pouring三个禁止事件的全片否定覆盖 |
| 新书本 | `task_review/gate_decision.json: needs_review` | open_book/move_box覆盖；`grasp_book`必要事件未建立 |
| 新柜门 | `task_review/gate_decision.json: needs_review` | open_left/remove_contents禁止事件覆盖 |
| 已有白碗修复视频 | `task_review/failure.json`及`task_review_protocol_v2/failure.json`均为`review_schema_violation` | 本批通用结构化审查尚无有效门控结果 |

这些是证据不足或协议失败，不等同于已经证明视频发生错误动作。白碗之前的助手/VLM候选语义通过观察仍保留，但不能替代本批失败的结构化契约审查。

因此，本批当时自动`task_equivalent_candidate=true`为0/4，正式质量发布为0，训练准入为0。对四条完整pair分别实际调用`validate_pair(mode="training")`，均以`Training requires quality_release_approved`拒绝。inspection没有静默提升批准状态。

计数口径：4个不同合成媒体内容，其中3个本批新生成、1个复用既有白碗修复视频；它们对应3个直接选用的真实人类窗口、4条现有机器人query和4条显式cross-demo边。本次新增真实人类录制0、新增机器人物理轨迹0、启动训练0。27项adapter测试已通过。后续若审核协议修复或人工质量判定改变，应另行保留审核产物并更新质量统计，不能倒改本次inspection快照。

### 最终审核更新：补证据与协议容器修复

以下更新仅重新核对新增审核JSON并追加本报告，没有重跑CPU读取，也没有修改上述`ac81e404ed146b5638ca`配对快照。

**书本抓取证据已补齐。** 新目录：

`/mnt/sdg/datasets/real_sim_human_pipeline/batch_20261020b/jobs/closed_book_to_open_box_891912000/task_review_dense_refinement_v1/`

`refinement_plan.json`保留原24帧，并在原第68–79帧间加入69–78共10帧，最终输入34帧。`result.json`为schema有效的新远端观察，`gate_decision.json`已匹配`grasp_book/lift_book/place_book/release_book/withdraw`；原`REQUIRED_EVENT_NOT_ESTABLISHED:grasp_book`不再出现。剩余两项仍是`FORBIDDEN_EVENT_COVERAGE_MISSING:open_book/move_box`，故仍为`needs_review`。这是一个实际的“定位缺失证据→增加局部帧→必要事件得到建立”案例；没有把禁止事件unknown改成正常，也没有证明所有未采样帧连续正确。

**白碗协议已完成有记录的离线容器归一化。** 新目录：

`/mnt/sdg/datasets/real_sim_human_pipeline/batch_20261020b/jobs/apple_bowl_transfer_891911000/task_review_protocol_shape_adapter_v1/`

`result.json`记录对`task_review_protocol_v2/provider_response.json`的8处`/events/event_000`至`event_007`执行`wrap_object_in_singleton_array`，各项包装前后内容hash相同；没有改动值、时间戳、证据或原始响应。原provider响应仍标`provider_schema_compliant=false`，派生结构通过schema验证，`provider_calls=0`。这不能写成原远端响应本来就符合schema。

其`normalized_evidence.json`与`gate_decision.json`现可正常产生，但仍为`needs_review`，保留三个禁止事件覆盖unknown，另有`FINAL_EVIDENCE_OUTSIDE_BOUNDARY:final_apple_supported`。该支撑观察同时包含frame136/5.666667秒和frame192/8.0秒；归一化把二者聚成`[5.666667,8.0]`，当前门控因区间起点早于末态窗口而提出问题。**这不是缺少最终帧，更不是已经证明末态苹果无支撑。** 后续应审查“跨时段稳定关系的报告区间”与“末态窗口必须覆盖的证据点”之间的表示规则；本轮没有为了放行而删去早期证据或放宽门槛。

最终四个候选都有可读取的规范化观察及门控结果，结果仍为4个`needs_review`；自动`task_equivalent_candidate=true`仍0/4，正式quality release仍0，训练准入仍0。上方初次审核的失败与缺失记录保留为历史，不覆盖。

### 本批审查总数与展示来源

仅统计`batch_20261020b/jobs/`内实际`request_metadata.json + call_started.json + provider_response.json`：**远端审查10次＝3次anchor比较＋1次手数专项＋6次视频审查**，不含视频生成或Nano编辑，也不把此前白碗修复研究的历史调用计入本批。连同3次Nano编辑与3次Seedance提交，本批共16个付费阶段请求。

| 审查类别 | 目录 | 远端次数 | 视频输入帧次 |
|---|---|---:|---:|
| anchor比较 | 新黑碗、书本、柜门各自的`anchor_review/` | 3 | 不适用，静态图比较 |
| 柜门手数专项 | `right_cabinet_open_only_891914000/anchor_body_inventory_v1/` | 1 | 不适用，单张静态图；输出最小2只且精确数量未知，未放行 |
| 四参考初次任务审查 | 四个job各自的`task_review/` | 4 | 96 |
| 白碗协议重审 | `apple_bowl_transfer_891911000/task_review_protocol_v2/` | 1 | 24 |
| 书本局部加密审查 | `closed_book_to_open_box_891912000/task_review_dense_refinement_v1/` | 1 | 34 |
| 白碗容器形状适配 | `apple_bowl_transfer_891911000/task_review_protocol_shape_adapter_v1/` | 0 | 复用上述24帧响应，不再调用模型 |

6次视频审查合计154个图像输入帧次，重复帧按实际重复输入计算，不能当作154个独立样本。最终每视频采用的最新观察分别为黑24、书34、柜24、白24帧；白来自明确记录的离线适配。CPU数据审查仍是另一个口径：4条显式pair、12次固定query读取。

用于最终`review_bundle`四行×四角色展示的真实源可直接按以下现有文件取值，无需改pair快照：

- `batch_20261020b/input_manifest.json`的每个job：`source_window`给出真实视频与精确窗口；`robot.video`给出sim donor原生512视频，`robot.source_episode_row_original`保留donor身份。
- `training_pair_inspection_snapshots/ac81e404ed146b5638ca/pair_manifest.json`的每条pair：`reference.video`给出实际合成视频，`query.episode_row.video`给出实际机器人query策略视频，`query.episode_row.reference_video`另给出同query的原生512展示流。
- 同快照的`fixed_sample_cpu_inspection.json`给出实际模型输入32帧索引、query帧和数值验证；展示流不能冒充模型实际使用的分辨率。

四行16个展示角色是对比栏位数；其中真实苹果窗口在两行重复，donor/query机器人也交叉复用。展示数量不得累加成16个独立录制或16条新的训练样本。
