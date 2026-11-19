# 可运行流水线：第一版

2026-10-19。代码已实现，CPU数据准备和离线协议/门控测试已运行；尚无真实API生成结果。它是首批可验证实现，不代表视频质量或下游策略收益已经证明。

## 当前交付

- [核心代码](r2h_pipeline/cli.py)：可恢复的SQLite任务记录、请求准备、提交、查询、下载、盲审、配对候选导出。
- [实际输入catalog](prepared_catalogs/long_train_two_groups_v2/catalog.json)：train中2组×3目标=6条，12条媒体完整解码与时间戳检查，96张事件覆盖PNG；原视频65.55–120.85秒，无裁剪/加速。
- [运行账本](pipeline_runs/bootstrap_v1/ledger.sqlite3)、[启动检查](pipeline_runs/bootstrap_v1/doctor.json)：首次启动未配置API凭据，没有网络请求。
- [准确API清单与接口限制](API_REQUIREMENTS.md)、[配置示例](config.example.json)、[凭据变量示例](.env.example)。

第一版已经实现：同初态任务导入、源/任务/提示词哈希绑定、完整阶段证据、同组共享且不含目标选择的锚帧请求、Seedance异步任务、Gemini匿名视频审核、事件/角色/时间门控、失败记录、有界提交、同split同目标不同布局配对。

尚未接通或验证：真实API账号/地区/模型配额、生成质量和审核阈值校准、SAM/CoTracker批量视觉核验、源深度/mask补录、分钟级视频连续扩展、Files API大视频上传、训练reader最终准入与策略收益。当前通过审核也仅得到配对候选，`training_admitted=false`。

## API准备

推荐两个平台、三个模型权限：

| 平台 | 模型 | 用途 |
|---|---|---|
| 火山引擎方舟 | `doubao-seedance-2-5-260628` | 视频生成/编辑 |
| Google Gemini API | `gemini-3.1-pro-preview` | 视频盲审与结构化事件提取 |
| Google Gemini API | `gemini-nano-banana-2.1` | 同组共享人类初始锚帧 |

在本机环境变量设置`ARK_API_KEY`和`GEMINI_API_KEY`，或将它们写入本目录`.env.local`（由用户自行填写，默认忽略，不入日志）。凭据文件只接受字面`NAME=value`，不会执行shell。Google两个模型采用同类key，但各模型权限需要账号实际开通。

统一中转若兼容上述原生协议，可以改config中的base_url和model。OpenAI兼容chat/completions协议并不自动兼容Gemini generateContent或Seedance任务接口；需根据中转文档另做适配。

## 命令

下面从工作区根目录执行。入口脚本也支持绝对路径及任意工作目录。

```bash
R2H_PY=/mnt/sdf/dataset/zhouxunzhe/zhiqi/PolicyGen/.venv/bin/python
R2H_CLI=/mnt/sdf/dataset/zhouxunzhe/zhiqi/data_generation_methods/scripts/run_r2h_pipeline.py
R2H_DB=/mnt/sdf/dataset/zhouxunzhe/zhiqi/data_generation_methods/pipeline_runs/bootstrap_v1/ledger.sqlite3
R2H_CFG=/mnt/sdf/dataset/zhouxunzhe/zhiqi/data_generation_methods/config.example.json

# 仅显示凭据是否存在，不调用API、不显示密钥。
"$R2H_PY" "$R2H_CLI" --config "$R2H_CFG" doctor
"$R2H_PY" "$R2H_CLI" --db "$R2H_DB" status
"$R2H_PY" "$R2H_CLI" --db "$R2H_DB" list
```

实际catalog已注册。需要另建小批次时用`prepare --source <正式policy_rows.json> --output <新目录> --groups 2 --split train`；同输出重复导入核对哈希并复用，无重复付费操作。

`list`输出的group和job ID可用于以下命令，占位符需替换：

```bash
# 无API：生成同组共享锚帧请求，不读取目标变体。
"$R2H_PY" "$R2H_CLI" --db "$R2H_DB" --config "$R2H_CFG" plan-anchor --group GROUP_ID

# 有API：图像模型调用。也可用bind-anchor接入其他方法产出的图像及provenance。
"$R2H_PY" "$R2H_CLI" --db "$R2H_DB" --config "$R2H_CFG" --env-file data_generation_methods/.env.local generate-anchor --group GROUP_ID

# 无API：按完整任务语义准备30秒生成请求；不是截原视频前30秒。
# 是否完整必须由生成后的事件审核判断，不能在准备时承诺。
"$R2H_PY" "$R2H_CLI" --db "$R2H_DB" --config "$R2H_CFG" plan --job JOB_ID --mode semantic --duration 30

# 有API：每次最多提交1条、审核1条、查询8条；一次遍历后退出，不无限等待。
"$R2H_PY" "$R2H_CLI" --db "$R2H_DB" --config "$R2H_CFG" --env-file data_generation_methods/.env.local work --max-submissions 1 --max-reviews 1 --max-polls 8

# 后续再次运行work即可继续查询；它不会重发submission_unknown或自动重试失败。
# 导出仍是候选，不写入原训练master，也不启动训练。
"$R2H_PY" "$R2H_CLI" --db "$R2H_DB" export-pairs --output data_generation_methods/pipeline_runs/bootstrap_v1/pair_candidates.json
```

同组只生成一次初始锚帧，再给三个目标复用。锚帧仍需要初态和最终视频质检；自动绑定不表示已经通过完整质量审核。正式大批量前应先逐组检查。

## 受控编辑分支

完整源片在4–30秒内时，可用`prepare-edit-media --source SOURCE.mp4 --output DERIVED.mp4`生成API副本：保持原时间轴，重采样24FPS并放大短边至768，保留源/派生hash及时间差检查。原动作时钟不改。对超过30秒的源直接拒绝，不截前缀。

Seedance原视频编辑需要可访问的视频URL或asset ID。将派生文件上传至自己的TOS等存储，使用签名URL或素材ID；当前工具不自动上传，也不自动开放文件为公网。以`plan --mode edit --video-uri <HTTPS或asset URI> --derived-media <DERIVED.mp4.json>`准备请求。URI与本地媒体的远端上传绑定尚需用户/上传端验证，不能在此版本宣称已自动校验远端字节。

目前6条多目标长片使用语义生成路线：首帧+完整事件说明→自然人类操作。若30秒容不下完整事件，审核拒收；连续扩展尚未实现，应另设计而不是自动压缩或降级标准。

## 盲审与发布边界

审核请求只发送生成视频、所有目标都相同的匿名候选与角色说明、JSON格式定义，不发送正确目标、生成prompt、源路径或任务名。私有任务定义在本地比对。

审核器要求完整初终态、对象身份、实际搬运动作、释放及目的地、时间证据。未知、低置信、额外操作、步骤缺失进入needs_review/rejected。当前置信阈值为试运行设置，尚未校准。没有像素跟踪工具时不能把这层语义审核称为物理验证。

`accepted_for_pairing`仍不等于训练准入：导出只建立同目标、跨layout、同split候选，并明确synthetic/no-human-3D-GT。后续须完成真实小试校准及训练reader验证。

## 恢复与费用边界

- 每条生成默认最多2次尝试；第二次需明确处理失败后重新plan，work不无限重试。
- POST超时或提交后落盘失败记`submission_unknown`，不重发。若在供应商控制台找到task ID，用`attach-task --job ... --task-id ...`恢复查询。
- 下载记录绑定task ID、请求及文件hash；未知已有文件拒绝复用；同任务的签名URL更新不会触发重复下载。
- 图像调用也先落独占调用标记；无保存结果的中断不会自动重发。
- 审核若进程在reviewing时被终止，当前需要检查保存的invocation/response再恢复；不通过自动重发掩盖不确定费用。
- 记录供应商usage；暂未做货币计价和账单核销。任务数预算不等于精确金额上限。

## 验证

```bash
"$R2H_PY" -B -m pytest -q -p no:cacheprovider data_generation_methods/tests data_generation_methods/r2h_pipeline/tests
```

测试包含真实小型CPU媒体准备，以及禁用网络的假供应商协议/超时/重复提交/哈希篡改/盲审/配对检查。假供应商只在临时测试目录使用，不能当生成质量或真实付费链路验证结果。
