# 条件触发修复规划器

`repair_router.plan_repairs(facts, verify_files=True)` 是独立的确定性规划模块。
它**不调用模型、不修图、不重新提交视频、不批准训练**；外层执行器执行返回的
`actions`，保存新的原始证据、更新调用计数，再传入新的 facts。

```python
from data_generation_methods.r2h_pipeline.repair_router import plan_repairs
plan = plan_repairs(facts)
# 外层只执行 plan["actions"] 中已实现的 kind，并保留调用幂等记录。
```

```bash
PolicyGen/.venv/bin/python -m data_generation_methods.r2h_pipeline.repair_router \
  /path/to/facts.json --output /path/to/repair_plan.json
```

## 输入

所有绑定均为 `{"path": "实际文件", "sha256": "64位小写十六进制"}`。
默认逐个复核文件内容；`verify_files=False` 仅供测试/检查，并禁用视频提交资格。

```json
{
  "schema": "r2h_repair_facts_v1",
  "reference": {"path": "真人来源视频", "sha256": "..."},
  "anchor": {"path": "当前人类初图", "sha256": "..."},
  "protocol": {"state": "ok"},
  "hidden_state": {
    "required": false,
    "reveal_condition_present": false,
    "condition_transport_verified": false
  },
  "actor_inventory": {
    "maximum_visible_hands": 2,
    "review_scope": "none",
    "review": null
  },
  "pair_validation": {"state": "not_run"},
  "attempts": {
    "body_reviews": 0,
    "anchor_repairs": 0,
    "contact_reviews": 0
  },
  "budget": {
    "paid_calls_remaining": 3,
    "max_body_reviews": 2,
    "max_anchor_repairs": 1,
    "max_contact_reviews": 1,
    "max_review_frames": 48
  }
}
```

补充字段：

- 已有候选视频时加入顶层 `video` 绑定。
- 隐藏状态必须由额外画面提供时，`required=true`。实际已有展示该状态的条件图后才可
  设置 `reveal_condition_present=true`，同时提供 `condition_artifact`。
  确认该图确实进入所选模型的实际条件载荷，才可设置
  `condition_transport_verified=true`，同时提供 `transport_evidence`。
  这两个布尔值属于外部执行器提供的事实；规划器校验绑定，不自行证明画面内容或
  provider 真正利用了条件。文字提到隐藏物体不算视觉条件传输证据。
- 手数 `review` 使用 `anchor_body_review.SCHEMA` 原始观测格式。
  有观测时需要 `review_scope=global|localized`、`media_sha256`、
  `evidence_artifact`。绑定的 JSON 必须等于该 review，或其 `review` 字段必须等于
  当前 review。局部复核仍未知时，只有明确
  `localized_review_can_add_evidence=true` 才会再规划付费复核。
- 接触未知时加入 `contact`，包含候选 `media_sha256`、原审查 `evidence_artifact`、
  `frame_count`、`base_frame_indices`、`unknown_event_ids` 和 `requests`。
  requests 遵循已有 `refinement.py` 格式：
  `{"query_id":"grasp","frame_interval":[68,79],"interval":"start_inclusive_end_inclusive"}`。
  该 query 必须出现在显式未知列表中。
- `pair_validation.state=passed|failed` 时必须绑定 `evidence_artifact`；
  验证本身仍由 `training_pairs` 完成。

## 条件与动作

规划顺序是协议恢复 → 隐藏状态 → 手数 → 接触 → 配对。
上游仍有阻断时，不提前支付下游生成或审查费用。

| 条件 | 返回的下一步动作 | 不能做什么 |
|---|---|---|
| 付费提交结果不明 | `reconcile_existing_submission` | 不自动重发 POST |
| 已有 task ID，状态 pending | `poll_existing_task` | 不重新生成 |
| 严格 schema 不通过且有可用无损适配 | `apply_lossless_shape_adapter` | 不补字段、不改真假或时间 |
| schema 变化不在无损范围 | `inspect_schema_failure` | 不把语义猜测当格式修复 |
| 必需隐藏状态没有条件图 | `extract_reveal_or_terminal_anchor` | 不提交首图单条件视频 |
| 有条件图，传输能力未经验证 | `verify_multi_anchor_condition_transport` | 不假设模型已接收图像；此动作不授权付费探测 |
| 尚未做手数审查 | `review_anchor_body_global` | 不把全图审查称为局部证据 |
| 已有手数未知，或全图报告超预算 | `review_anchor_body_localized` | 不先修图掩盖观测不确定性 |
| 局部复核确认多余手 | `repair_anchor_body` | 不超过编辑次数；新图必须重新审查 |
| 局部复核未知且无新证据 | `request_manual_body_review` | 不对同样输入循环收费 |
| 接触未知且有短区间 | `review_dense_contact_frames` | 不把选帧直接当接触证明 |
| 连续区间放不进帧预算，或没有新增帧 | 保留未知，延期 | 不用稀疏抽帧冒充连续证据；不收费 |
| 配对校验失败 | `repair_explicit_pair_manifest` | 不改动作真值来强行通过，不交付训练 |

`paid_calls_reserved` 只是本计划拟使用的预算，**不是实际费用或已发生调用**。
实际调用前，外层执行器仍需执行已有 `paid_stage` 的幂等保护、记录 started/result，
并将真实累计次数传入下一轮。规划器不维护跨进程状态，不能替代 durable journal。

`ready_for_submission` 只代表这里检查的生成前置条件已满足；
隐藏状态必需时 `first_frame_only_submission` 始终为 false。
`candidate_inspection` 只代表可以继续核对任务证据和配对。
所有输出固定 `actions_executed=false`、`quality_release_approved=false`、
`training_admitted=false`，不存在“修复规划通过即批准训练”。

## 当前接入边界

本模块**不自行改动旧 batch runner**。新增的柜门案例执行器
[`execute_conditioned_pipeline_20261020c.py`](../scripts/execute_conditioned_pipeline_20261020c.py)
已经把部分动作接到真实 handler：

| 路由动作 | 执行器实际执行 |
|---|---|
| `extract_reveal_or_terminal_anchor` | 调用 `edit_first` 复用已有初图缓存，再调用 `edit_last`，由原仿真终图编辑出揭露内部物体的人类终图 |
| `verify_multi_anchor_condition_transport` | 用与实际 video POST 同一个 `build_video_payload` 形成首尾图载荷，执行 `prepare_submit` 本地校验，保存 roles / payload hash / 两图绑定 |
| `review_anchor_body_global` | 调用真实四静态图 `review_conditions`，把原始手数/不确定性/位置无损映射到手数模块，保留原请求和回复绑定 |
| 视频前置条件通过 | 另核验全部 `condition_gate` 与 prepared payload hash 后，调用 `video` 实际提交并按已有 task ID 轮询 |

载荷准备只证明图像条件进入本地构造的请求结构；它明确记录
`provider_condition_use_verified=false`。生成模型是否正确利用该条件，需要对实际视频再审查。
四静态图审查记录为 `global`，没有冒充局部裁剪审查。

执行器默认 `--plan-only`，只有显式 `--execute` 才能调用已有幂等付费阶段；
默认最多4个付费阶段，包含此前已执行的初图阶段。隐藏状态的“提取条件”在通用规划器里
是本地动作，但此案例还包含模型编辑终图，因此执行器会再次按实际缺少的 paid stage
检查预算，不能借本地动作之名绕过付费预算。

每一步将事实、计划和执行 journal 保存在案例目录的 `dispatcher_v1/`。
`action_started` 不是执行成功；只有 handler 完成后才记 `action_executed=true`。
未实现的动作（例如本案例的局部手数复核、再编辑初图、接触审查自动执行）会明确停止，
不绕过未知。接触补帧工具仍由既有审核入口调用。

如需运行此固定案例：

```bash
PolicyGen/.venv/bin/python data_generation_methods/scripts/execute_conditioned_pipeline_20261020c.py --plan-only
# 实际执行由主操作者明确启动：
PolicyGen/.venv/bin/python -u data_generation_methods/scripts/execute_conditioned_pipeline_20261020c.py \
  --execute --max-actions 6 --max-total-paid-stages 4
```

外层执行器仍需在每次产物变化后重新绑定证据。
未实现的动作继续阻断，不得把“已返回计划”写成“已执行修复”。
局部可见手数量不是人体姿态真值；连续 RGB 审查也不是接触力或三维真值。
