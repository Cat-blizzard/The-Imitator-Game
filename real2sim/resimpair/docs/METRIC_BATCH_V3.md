# 米制资产批量候选入口 v3

这是可执行候选入口，尚未替换旧默认管线。`scripts/batch_pickplace.sh metric-v3` 保留所有旧子命令。

一次 `prepare` 完成：

1. 读取明确指定、不可变的最终资产 BUNDLE 与严格容器 catalog，校验两份 consumer 报告及所有绑定；不允许退回未过滤的旧资产池。
2. sampler v4 随机选择同一划分内的资产、离线尺寸版本、容器与干扰物；目标数量可为 3–8。任务身份绑定完整生成条件，编译后另外保存实际场景指纹。
3. 按目标所在机械臂一侧提出源物体位置，经过原桌面间距和容量检查，再检查同一机械臂与同一缓存抓点的局部接近、抬起、下放和退出条件。
4. 保存每次失败提案、任务合同、初始场景、资产与 pose 绑定；核对目标、容器、干扰物的原始来源划分及任务 ID。
5. 创建 driver4 执行批次，使用已有 128 次 CPU 物理累计账本。`prepare` 不运行物理，也不调用 API。

正常配置使用已验证的 `checked_withdrawal_v3` 退出动作。只有在 `prepare` 明确加入 `--experimental-closure` 时才绑定 driver6 闭合实验；此开关不会晋级为默认，也不能在 `run` 临时更换已冻结驱动。

本轮已发布输入：

- `runs/semantic_scale_consumer_20261030_v1/final_consumer_v2/BUNDLE.json`：935 个原始来源、941 个版本、26634 个 pose。它是语义/尺度过滤后的缓存候选池，不是 935 个已物理抓放成功资产。
- `runs/container_affordance_consumer_20261030_v2/sampler_catalog.json`：159 个容器版本、148 个来源，经过原支撑规则重检。几何资格不保证任意物体组合都能放入或完成整条任务。

从仓库根目录执行以下准备命令即可复用现有账本。`--name` 必须是尚未准备过的新批次名；失败后先检查原目录，不换名字掩盖失败。

```bash
scripts/batch_pickplace.sh metric-v3 prepare \
  --campaign runs/production_integration_20261030_v1 \
  --name unified_metric_v3_next \
  --bundle runs/semantic_scale_consumer_20261030_v1/final_consumer_v2/BUNDLE.json \
  --container-catalog runs/container_affordance_consumer_20261030_v2/sampler_catalog.json \
  --scenes 2 --count-schedule 3,8 --seed 2026103050 --goals all
```

也可以不传 `--count-schedule`，在 `--min-objects 3 --max-objects 8` 内按场景 seed 采样。`--goals` 可选择 `select_one`、`select_other_bin`、`collect_subset`、`distribute_groups`、`collect_all`。`all` 表示保留容量允许的任务，并记录不满足容量的任务；不是强行把所有物体塞进一个容器。

准备完成后，以下命令执行已经冻结的批次，最多新增一次物理尝试：

```bash
scripts/batch_pickplace.sh metric-v3 run \
  --cohort runs/production_integration_20261030_v1/unified_metric_v3_next \
  --limit 1
```

执行直接复用原 driver 的全局锁、campaign 锁、480 秒超时、128 次累计上限和一次提交记录。原 reservation 没有 terminal 时会停止并要求恢复原作业，不会再次提交；不会创建新预算。最终物理审核仍由独立验收入口完成，本入口不修改或放宽规则。

如果 CPU 编译已完成，但其他执行器占用 campaign 锁，原提案和 `prepared_cases.json` 会保留。待锁释放后提交原编译结果，不重新抽样：

```bash
scripts/batch_pickplace.sh metric-v3 commit \
  --campaign runs/production_integration_20261030_v1 \
  --name unified_metric_v3_next
```

实际验证在 `unified_metric_v3_N3_N8_r1`：CPU 采样及编译用时 41.52 秒，2 个场景、8 个任务。N=3 场景位于 development，含 3 个目标与 3 个干扰物；N=8 场景位于 heldout_asset，含 8 个目标与 5 个干扰物。所有原始来源划分和任务 ID 校验通过。两场景都因单容器容量不足明确不提供 `collect_all`，各保留其余 4 类任务；6 次局部路径条件失败的提案也已保存。没有执行这 8 条物理轨迹。

5 项入口回归通过，包含拒绝旧未过滤池、拒绝错误容器 catalog、实际 N=3/N=8 来源和 ID 检查，以及默认只派发 driver4。原采样日志保留了一条 trimesh 数值转换警告，其确切来源尚未定位；不可由编译通过推断动态物理成功。完整路径碰撞、夹持稳定性、容器内放置稳定性仍需实际执行验证。
