# 家居资产与受控布局候选：2026-10-20 E

已完成 **12 类本地家居资产的 CPU 预检、24 个新场景配置候选**。首次实跑 8 配置为 5 个 native 通过、3 个初态冲突；修复并换用 strict-v3 检查后，8 个配对家居场景配置均通过 native 验证，实际覆盖 **8 类新增家居资产**。另完成 2 个相机事件可见性修复配置。原始失败与旧 4 个 pilot 产物保留，训练未启动。

去重后，当前全部 strict-v3 结果是 **15 次成功执行、14 个视觉场景配置、6 组不同的精确动态 rollout**；其中这批 8 家居配置只包含 2 组动态 rollout。杂物与相机增广是合法的不同观察，不能算成 15 条独立运动数据，也不能把上述工程资格测试当成总体成功率。

## 产物与入口

- 有效资产版本：`/mnt/sdg/datasets/real_sim_human_pipeline/expansion_20261020e/household_assets_v3/`
  - `catalog.json`：12 个完整资产规格。
  - `asset_gallery.jpg`：CPU 几何/顶点采样色的粗略预览，不能替代实际 UV 纹理和 PBR 渲染检查。
  - `<category>/visual.glb`、`collision.obj`、`asset.json`、`source_record.json`、`preview.jpg`。
- 原始布局候选版本：`/mnt/sdg/datasets/real_sim_human_pipeline/expansion_20261020e/household_layouts_v2/`
  - `candidate_manifest.json`：24 个配置、拒绝原因、来源绑定。
  - `first8_candidate_configs.json`：首批 8 个配置的冻结原始输入。
  - `cases/*.json`：可单独交给 strict adapter 的场景配置。
- 修复后 8 配对配置：`expansion_20261020e/household_layouts_v3/proposals.json`。
- 当前独立实证统计：`expansion_20261020e/household_actual_qualification_strictv3_v1.json`。
- 代码：`human_to_robot_pipeline/household_assets.py`、`expand_household_cases.py`。
- 实际覆盖审计：`human_to_robot_pipeline/audit_household_coverage.py`。
- 原生非目标物反馈修复：`human_to_robot_pipeline/repair_household_layouts.py`。
- 分阶段汇总、相机原始数组复核与去重：`human_to_robot_pipeline/summarize_household_qualification.py`。
- 验证：`tests/test_household_assets.py`。

早期 assets v1/v2、layouts v1 保留为调试历史。v3 修正了遥控器选择、订书机朝向与 CPU 顶点采样颜色索引，但仍未实现每像素 UV/PBR 渲染；layouts v2 调整类别顺序，优先让明显可辨的家居物体进入最先两组，并覆盖全部 12 类。

## 资产从哪里来

复用已下载的 `/mnt/sdg/datasets/open_assets/household_scaleup_20261020_v1/`，本轮只读取本地 Objaverse GLB。选中记录包括原始 UID、名称、作者、来源 URL、CC-BY-4.0 许可、源文件 SHA256。上游 catalog 是持续追加日志，故单独冻结被选记录及其内容 hash，不能将它当成静态索引。

本轮为每类选择一个 mesh，共 12 个 mesh，并非每类已有大量几何实例。

| 类别 | 设计尺寸 X×Y×Z，厘米 | 碰撞近似 |
|---|---:|---|
| 木制杯 mug | 6.73×10.00×9.26 | 凸包 |
| 易拉罐 soda_can | 6.00×6.00×11.00 | 保守包围盒 |
| 水瓶 water_bottle | 5.14×5.14×16.00 | 保守包围盒 |
| 带盖果酱罐 closed_jam_jar | 9.35×9.23×10.00 | 保守包围盒 |
| 纸巾盒 tissue_box | 14.00×6.15×7.00 | 凸包 |
| 皂液瓶 soap_dispenser | 5.34×6.35×14.00 | 保守包围盒 |
| 锅铲 spatula | 18.00×4.09×2.47 | 保守包围盒 |
| 遥控器 remote_control | 3.61×14.00×1.27 | 凸包 |
| 订书机 stapler | 6.53×11.00×2.08 | 凸包 |
| 螺丝刀 screwdriver | 8.32×16.00×2.45 | 保守包围盒 |
| 水壶 kettle | 16.00×12.87×13.22 | 凸包 |
| 纸盒装奶 milk_carton | 4.61×6.49×15.00 | 凸包 |

这些尺寸是显式设计选择，**不是从人类视频恢复出的真实尺寸，也不是原作者保证的测量值**。保留 `source_to_adapter_4x4`、统一缩放、方向假设、偏移量。glTF +Y 转成仿真 +Z；扁平工具额外旋转到最短轴朝上；XY 包围盒中心归零、底部 Z 归零。相同变换同时烘焙到视觉与碰撞文件，runtime 不得再次缩放。

每个源文件验证 SHA、顶点/法向有限值、非退化尺寸；重新加载导出的视觉/碰撞后比较包围盒。碰撞使用完整凸包；若顶点超过 192，退回包含整个物体的包围盒，不能通过删掉凸包极端顶点来获得看似简洁但漏碰撞的几何。全部碰撞体满足封闭、凸、正体积。

这批物体只允许作为 **静态、禁止触碰的干扰物**。凸包/包围盒会填平杯把、空腔等，不能据此支持杯把抓取、倒液体、开盖等交互。它们必须同时进入 renderer、physics collision 和 planner world；只渲染不加碰撞不合格。

CPU 目检发现：订书机像透明扁盒，语义辨识度较差，因此不进入最先两组；木杯风格化，不能称为照片级真实。目检没有替代真实渲染、物理或规划验证。

后续实际渲染又发现 milk_carton 的黑白印刷在当前视角和分辨率下不易辨识。检查原始 GLB 的 base-color atlas，里面本身包含大面积黑白几何图案与斜排 MILK 字样，**没有证实渲染器、法线或几何损坏**。原纹理与真实 frame0 局部对照位于 `expansion_20261020e/milk_material_review_v1/atlas_vs_actual_crop.png`，原 atlas PNG 直接抽取自 GLB 的 image bufferView，视频、帧号、crop 像素坐标和 SHA 均保留。

旧 CPU 预览只按顶点取色，漏掉了面内 UV 纹理，不能用来证明贴图真实效果；新版标题与元信息已改为“粗略顶点色几何预览”。可选的 `repair_household_material.py` 仅生成人工哑光浅色纸盒 PBR 工程 variant，GLB 的 BIN、mesh、UV、node transform 与碰撞体保持不变。它不是模型 bug 修复，也不是恢复的真实材质；是否使用等待实际渲染和可读性复核。原材质与旧成功物理记录继续保留。

## 24 个配置如何生成

沿用已审核 apple-transfer task contract，固定“苹果从已占用源碗转移到空目标碗”的目标与人类来源。4 组新布局使用 seed：

| 组 | seed | active arm | 人类窗口 ID |
|---|---:|---:|---|
| g00 | 2026302001 | 0 | ego_part2_basic_pick_place_2174 |
| g01 | 2026302003 | 1 | ego_part2_basic_pick_place_6057 |
| g02 | 2026302017 | 0 | ego_part2_basic_pick_place_8006 |
| g03 | 2026302018 | 1 | ego_part2_basic_pick_place_2174 |

每组 6 个变体：

| 配置 | 杂物 | 相机 | 外观 | 当前状态 |
|---|---:|---|---|---|
| baseline | 0 | 基础 | native seed 默认 | g00/g01 strict-v3 native 通过；g02/g03 未跑 |
| clutter2 | 2 | 基础 | 同 baseline | 同上；g00 使用已记录的位置修复 |
| clutter4 | 4 | 基础 | 同 baseline | 同上；g00/g01 使用已记录的位置修复 |
| camera_only | 0 | 小幅扰动 | 同 baseline | g00/g01 实际 K/E 核验并 native 通过；可见性另查 |
| appearance_only | 0 | 基础 | 显式材质候选 | **blocked：override 尚未实现** |
| combined | 4 | 小幅扰动 | 显式材质候选 | **blocked：override 尚未实现** |

同组严格保持 native seed、目标物位置、人类窗口、split 相同。clutter2 是 clutter4 的前两项，不重新采样；布局/杂物/相机/材质候选使用独立 RNG stream。跨组改变布局的同时 native seed 自带的桌面、墙和 HDRI 可能变化，因此跨组差别不能解释成单独位置因素。

重要限制：`appearance.material_seed` 尚未被 adapter 消费。准备的 Poly Haven CC0 `angli_veneer`、`bamboo_wall_03` 材质是带 SHA 的文件候选；没有声称当前已经实现独立背景随机化。对应 8 个配置显式 blocked，首 8 资格配置不包括它们。

g00/g01 的基础和扰动相机已经用实际 `qa_visibility` 内外参核验；g02/g03 的候选范围尚未运行。相机变化是否执行必须由实际 intrinsic/extrinsic 证明，不能仅凭 JSON 有字段就确认。后续另外两个事件修复视角的实证见下文。

所有人类窗口与新布局均是 train；同一个人类来源派生的布局不能跨 split。此轮是训练场景资格候选，不是 unseen 测试集。

位置选择先检查目标物布局、保守 reach、物体包围球间隔、搬运中心通道、关键事件视锥、相机至事件的遮挡代理。4 个新布局 seed 被拒绝：3 个因 4 干扰物几何在有界搜索内放不下，1 个越过保守可达范围。没有为了保留它们而缩小资产或放宽门槛。保留的 24 配置均通过这一粗筛，但粗筛不能替代完整规划与物理验证；当前只有 g00/g01 的 8 个修复配对配置进入了 strict-v3 实跑。仍不宣称形式化连续碰撞安全，RGB 事件可见性、任务等价性与资产可读性也需单独复核。

优先 8 个 case ID：

```text
household_e_g00_baseline_2026302001
household_e_g00_clutter2_2026302001
household_e_g00_clutter4_2026302001
household_e_g00_camera_only_2026302001
household_e_g01_baseline_2026302003
household_e_g01_clutter2_2026302003
household_e_g01_clutter4_2026302003
household_e_g01_camera_only_2026302003
```

先以不改动的旧 4 seed 检验新 strict planner，再跑 g00 baseline→clutter2→clutter4。若 native collision/physical verifier 不通过，停止扩增并修复。相机组后置；背景 override 实现前不跑被阻塞的材质组。

## adapter 字段约定

`added_distractors` 每项包含 `instance_id`、`asset_uid`、`category`、`center`、`radius_m`、`yaw_rad`、`visual`、`collision`、`asset_files`、`must_remain_untouched=true`。`center` 是世界坐标下碰撞包围盒中心，单位米；文件本身已经按米归一、朝向烘焙，视觉和碰撞共享 frame。runtime 只围绕共用 bbox 中心应用 yaw 与平移，不能另加 scale。

`radius_m` 是保守包围球代理，不是真实接触几何。strict planner 应读取已加载 actor 的真实 world collision mesh，连同未选任务物体、源/目标碗、两只机器人和被抓持苹果一起检查。

## 复现

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PolicyGen/.venv/bin/python \
  -m data_generation_methods.human_to_robot_pipeline.household_assets \
  --output /mnt/sdg/datasets/real_sim_human_pipeline/expansion_20261020e/household_assets_NEW

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PolicyGen/.venv/bin/python \
  -m data_generation_methods.human_to_robot_pipeline.expand_household_cases \
  --assets /mnt/sdg/datasets/real_sim_human_pipeline/expansion_20261020e/household_assets_v3/catalog.json \
  --output /mnt/sdg/datasets/real_sim_human_pipeline/expansion_20261020e/household_layouts_NEW

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PolicyGen/.venv/bin/python \
  -m pytest data_generation_methods/tests/test_household_assets.py -q
```

输出目录必须不存在，避免覆盖旧证据。以上命令只做 CPU 资产和场景准备，不启动仿真、训练或付费 API。

## 实际反馈：原生非目标水果遗漏

主流程完成首 8 配置的小样本 strict 实跑。`household_layouts_v2/actual_coverage_snapshot_v1.json` 是只读快照：8 次尝试、5 个完整通过的证据链，3 个在 0 帧被初态保护拒绝。这不是无偏总体成功率，也不能只报告通过样本。

- g00 baseline 与 camera_only 通过；clutter2、clutter4 因纸巾盒和原生未选水果冲突失败。
- g01 baseline、camera_only、clutter2 通过；clutter4 因遥控器和原生未选水果冲突失败。
- **这份首轮快照当时**通过的带杂物样例只覆盖 milk_carton、spatula，不能把配置中的 12 类都当成已经物理验证；后续 strict-v3 扩大到 8 类，见下文。
- 原始内外参验证 camera_only 确实执行：g00 实际 FOV 约 1.30370，g01 约 1.32773，基础为 1.30000 rad；最大外参绝对误差约 1.4e-7。这里验证的是相机标定，不自动证明关键接触看得清。
- 这份首轮快照的完整通过样例覆盖 2 套 runtime 墙面/桌面/HDRI 组合，是 native seed 产生的结果，不代表 `material_seed` 或独立背景 override 已实现。

根因是候选粗筛只包括新杂物和目标碗，没有原生场景另一水果的实际占据范围。原先“剩余原生物体仍需 exact check”的限制确实变成了实际失败，不能留在文字里不改模块。

修复采用已成功同 seed baseline 的 `plans/unselected_object_world.obj`；该文件是 settled 初态实际碰撞 mesh，附输入、结果、receipt 和 mesh SHA。预设 seed 能确定名义布局，但不能替代动力学 settling 后位姿。新流程先有一个无杂物 baseline，再以实际原生 mesh 提出杂物；缺少此证据的冷启动粗候选显式携带 `native_unselected_dynamic_geometry_not_in_CPU_screen` blocker。

`household_layouts_v3/proposals.json` 保留同一任务、目标碗位置、seed、相机、资产、yaw，只改变必要杂物的 XY：

| 物体 | 移动 ΔX / ΔY | 原因 |
|---|---:|---|
| g00 纸巾盒 | +30.04 / −30.60 cm | 原位置与未选水果重叠；固定 yaw、其他杂物和保守桌面/搬运通道条件下，需要移到另一处空位 |
| g01 遥控器 | −4.16 / 0 cm | 原位置与未选水果重叠 |

纸巾盒总移动 42.88 cm，是明显的布局修正，不能描述成微小扰动。以上是 5 mm XY 网格里、固定其他变量的最小可行平移，不是全局最优布局。修复用 2.5 cm 轴向 AABB 分离余量，strict adapter 的原 2 cm 初态门槛完全不变；新 runtime 仍须重新核验，不能将 baseline mesh 推断当成新轨迹成功。

只有 3 个 case 真正改了几何，`changed_geometry_case_ids` 列出 g00 clutter2/clutter4 与 g01 clutter4，名称加 `_nativeclear_v3`。随后由于 strict-v3 增加了真实被抓持物体对场景的碰撞检查，主流程对完整 8 条配置重新回归。这些重复是验证新检查的执行，不计作新布局或独立运动。旧失败不删除，修复后作为新尝试保留。

实际覆盖审计按 `input.json` 精确匹配 case、绑定 receipt/qa 的 SHA、读取 `camera_from_world` 与 `intrinsic_224/512`、验证 runtime 材质文件 hash、核对实际静态杂物数量。缺少结果/标定/材质的地方保持 unknown，不用请求 JSON 代替完成证据：

```bash
PolicyGen/.venv/bin/python \
  -m data_generation_methods.human_to_robot_pipeline.audit_household_coverage \
  --proposals /mnt/sdg/datasets/real_sim_human_pipeline/expansion_20261020e/household_layouts_v2/proposals.json \
  --results /mnt/sdg/datasets/real_sim_human_pipeline/strict_20261020e \
  --output /mnt/sdg/datasets/real_sim_human_pipeline/expansion_20261020e/household_layouts_v2/actual_coverage_NEW.json
```

v3 修复集使用其自己的 `proposals.json` 单独审计。现在 8 个新 ID 都有各自真实执行的 result、receipt、原始数组和视频；没有将 v2 的成功记录复制成 v3 新结果。

## strict-v3 实际资格结果

独立读数产物是 `household_layouts_v3/actual_coverage_strictv3_v1.json` 和 `household_actual_qualification_strictv3_v1.json`。核对了 proposal、receipt、arrays/video/qa 的 SHA、actual camera K/E、runtime 材质和静态物体数量，8 条全部具有完整证据链：

| 布局组 | 配置 | 新家居杂物数 | 帧数 | native 验证 |
|---|---|---:|---:|---|
| g00 | baseline | 0 | 305 | 通过 |
| g00 | clutter2 | 2 | 305 | 通过 |
| g00 | clutter4 | 4 | 305 | 通过 |
| g00 | camera_only | 0 | 305 | 通过 |
| g01 | baseline | 0 | 277 | 通过 |
| g01 | clutter2 | 2 | 277 | 通过 |
| g01 | clutter4 | 4 | 277 | 通过 |
| g01 | camera_only | 0 | 277 | 通过 |

8 条记录的新增静态物体接触力最大值均为 0 N。实际新增资产类别为：

`soda_can、tissue_box、water_bottle、soap_dispenser、milk_carton、spatula、remote_control、closed_jam_jar`。

木杯、水壶、螺丝刀、订书机这 4 类目前仍只有 CPU 预检和候选配置，不能合并宣称“12 类全部完成物理/渲染资格”。已有 8 类也不等于全都达到高质量视觉门槛，例如原 milk_carton 在当前视角的可读性限制仍保留。

## 同轨迹相机修复：动作不变，关键时段更可见

原 g00 baseline 的目标在 `close` 标记的 16 帧内，达到 16 个可见 segmentation 像素的帧数是 0；整体平均图像质量通过没有保证这个关键局部时段可见。主流程在同 seed 上另测 opposite_low、opposite_high 两个视角，均 native 通过。

独立从原始文件重新检查了 `strict_20261020e/event_camera_repair_audit.json`：

- 三次执行均为 305 帧。
- `arrays.npz` 的 **18 个字段全部逐元素完全相同**；其中 14 个数值字段最大绝对差均为 **0**，包括命令、机器人状态、物体状态、TCP、时间戳和初始物理状态；布尔 mask/grasp 和 stage 序列也相同。
- 原始 camera extrinsic 发生变化、intrinsic 仍匹配请求，证明确实改了视角。

| 视角 | close 帧数 | close 中 ≥16 可见像素的帧数 | close 最少可见像素 |
|---|---:|---:|---:|
| 原 baseline | 16 | **0 / 16** | 0 |
| opposite_low | 16 | **16 / 16** | 92 |
| opposite_high | 16 | **16 / 16** | 50 |

两个修复视角的 grasp_reach、lift、release_above_rim、open 标记窗口也全部达到这一可见像素阈值。这证明可以在不改变该条实际动态 rollout 的情况下改善事件窗口中的目标可见性；**不证明仅靠像素数就看清了真实接触**。`close` 是控制器阶段标签，16 像素是本地诊断阈值，不是通用质量 GT。这两个相机是发现遮挡之后的工程修复，不是随机总体评测。

## 去重后的数据量：不要把观察增广当成独立运动

对 `strict_v3_motion_duplicate_audit.json` 中的 15 次 strict-v3 成功执行重新读原始数组，以 `native_commands + states + object_states + timestamps_seconds` 的 dtype、shape、内容做精确分组，复现了父流程的同一组划分：

| 层级 | 数量 | 说明 |
|---|---:|---|
| strict-v3 成功执行记录 | 15 | 4 个旧 seed 回归 + 8 个家居配对配置 + 1 次重复 g01/clutter2 回归 + 2 个相机修复 |
| 不同视觉场景配置 | **14** | 重复 g01/clutter2 不增加一个配置 |
| 不同精确动态 rollout | **6** | 4 个旧 seed 各 1；新 g00 共用 1；新 g01 共用 1 |
| 8 个家居配对配置内的动态 rollout | **2** | 两个 seed，各自的杂物/视角变体命令和状态完全相同 |

g00 的 6 个视角/杂物变体共用一组动态数据；g01 的 5 次执行共用另一组，其中包括重复回归。场景杂物和观察视角仍然可以增加训练中的视觉变化，但没有增加独立运动轨迹。

所有同 motion、同源人类 session 的变体必须留在同一 split，不能逐帧随机划分，也不能把同轨迹换视角放到另一 split 来报告泛化。当前 24 候选、8 家居配对配置、14 总视觉配置、6 独立动态组是不同统计口径，必须分别陈述。训练与生产接纳依然为 false。
