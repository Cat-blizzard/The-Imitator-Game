# 生成与审核 API：模型、请求边界与当前实现

2026-10-20 更新：已实测用户指定 Echoflow 中转原生 Gemini/Nano 接口，Seedance 超时恢复候选视频已完成但质检拒收。当前调用证据和方法边界见 [本轮实施计划](REAL_SIM_HUMAN_EXECUTION_PLAN_20261020.md)、[中转配置](config.echoflow.example.json)；下文2026-10-19的“未调用”是当时状态。

核验日期：2026-10-19。只查官方文档；没有读取真实密钥，没有调用付费模型。
这里的“文档支持”不代表当前账号已经开通模型，也不代表生成质量经过验证。

## 1. 本轮使用的明确模型

| 用途 | 配置中的准确模型 ID | 环境变量 | 状态与证据 |
|---|---|---|---|
| 机器人视频编辑为人类参考 | `doubao-seedance-2-5-260628` | `ARK_API_KEY` | 官方模型列表与 Seedance 2.5 教程列出该 ID 和视频编辑能力；账号需开通模型服务。[模型列表](https://console.volcengine.com/ark/region:cn-beijing/docs/model-list)、[教程](https://docs.volcengine.com/docs/ark/seedance-2-5?lang=zh) |
| 匿名视频语义审核 | `gemini-3.1-pro-preview` | `GEMINI_API_KEY` | 官方列出视频输入、文本输出和 structured outputs。它是 **Preview**，不是稳定版，也不宣称它是所有当前模型中最强的。[模型页](https://ai.google.dev/gemini-api/docs/models/gemini-3.1-pro-preview) |
| 可选的共享人类锚帧编辑 | `gemini-nano-banana-2.1` | `GEMINI_API_KEY` | 官方列为 Stable，支持图像输入/编辑、图像与文本输出；**不支持 structured outputs**。[模型页](https://ai.google.dev/gemini-api/docs/models/gemini-nano-banana-2.1) |

两个 Gemini 模型使用同一种 Google AI API key 机制；同一个项目的 key 可以作为两者凭据。
实际模型可用性、地区、配额、计费或权限仍由该项目决定，不能仅凭“有 key”推断均已开通。
Seedance 的当前开通条件见官方创建任务页“模型开通条件”。

## 2. Seedance：异步任务与素材限制

Base URL 为 `https://ark.cn-beijing.volces.com/api/v3`。
`POST /contents/generations/tasks` 返回 `id`；
`GET /contents/generations/tasks/{id}` 查询一次状态，成功后取 `content.video_url`。
鉴权是 `Authorization: Bearer <key>`。[官方创建任务 API](https://console.volcengine.com/ark/region:cn-beijing/docs/create-video-generation-task-api)、[官方调用流程](https://docs.volcengine.com/docs/ark/video-generation-tutorial?lang=zh&redirect=1)

受控编辑分支用显式 `omni_reference_task_type="edit"`、`ratio="adaptive"`、`duration=-1`。
当前长程小样的语义生成分支则用共享人类锚帧 `first_frame` 加完整任务说明，不上传超长原视频。
视频内容项是 `type="video_url"`、`role="reference_video"`；
视频来源只能是可访问的公网 HTTP(S) URL 或 `asset://...`，**不是本地路径，也没有文档支持 `data:video/...`**。
图片可用公网 URL、`data:image/...;base64,...` 或 asset。
共享锚帧配合视频输入时用 `role="reference_image"`；`first_frame/last_frame` 模式与多模态参考模式互斥。
上述限制位置：创建任务页“图片信息”“视频信息”“omni_reference_task_type”。

Seedance 2.5 编辑输入为 4–30 秒、24–60 fps，MP4/MOV，单视频不超过 200 MB；
视频像素数需在 407696–8295044，不能只看最长边是否达到 768。
输出支持 480p/720p/1080p。输入派生文件与原始机器人轨迹分别保存：
改变 API 输入的 fps 或分辨率不能重写原生动作时钟。
官方还说明编辑后时长可能与输入有约 0.4 秒差异；因此 `duration=-1` **不等于逐帧运动保持**。
素材尺寸、时长、解码和生成后的事件核验由 pipeline 其他环节完成，provider 不下载输入 URL 来验证它们。

最小请求结构（占位 URL，不能原样提交）：

```json
{
  "model": "doubao-seedance-2-5-260628",
  "content": [
    {"type": "text", "text": "视频编辑：保留物体与动作顺序，将机器人替换为人类双手。"},
    {"type": "video_url", "video_url": {"url": "https://YOUR_MEDIA_HOST/input.mp4"}, "role": "reference_video"}
  ],
  "omni_reference_task_type": "edit",
  "ratio": "adaptive",
  "duration": -1,
  "resolution": "720p",
  "generate_audio": false
}
```

## 3. Gemini：匿名 inline 视频 + JSON 审核

Base URL 为 `https://generativelanguage.googleapis.com/v1beta`，
使用 `POST /models/{model}:generateContent` 和 `x-goog-api-key` header。
视频 bytes 放入 `contents[].parts[].inlineData.{mimeType,data}`；
`generationConfig.responseMimeType="application/json"`，可同时指定 `responseJsonSchema`。
[REST 接口与字段](https://ai.google.dev/api/generate-content)、[视频输入](https://ai.google.dev/gemini-api/docs/video-understanding)

当前视频文档的概览表写 inline <100MB，而同页详细说明仍写总请求 <20MB。
实现采用保守的**整个序列化请求不超过 20 MiB**，单个原文件先限制 15 MiB；
Base64、prompt、schema 都计入总量。超限抛 `inline_limit_exceeded`，不截视频、不压缩、不静默上传。
Files API 暂未实现。

`review(video_path,prompt,response_schema)` 只发送所提供媒体、匿名 prompt 和 schema；
不追加本地路径、文件名、目标任务 ID、生成 prompt 或预期答案。
schema 是格式约束，语义是否满足由独立 `review.py` 核验。
响应截断、拦截、非 JSON 对象均拒绝；不通过“补齐 JSON”或剥代码块掩盖失败。

`generate_image(image_path,prompt,model="gemini-nano-banana-2.1")` 使用同一端点，
请求 `responseModalities=["TEXT","IMAGE"]`，**不设置 JSON schema**；返回 Base64 图像数组，
由调用方保存并记录来源。[图像生成文档](https://ai.google.dev/gemini-api/docs/image-generation)

## 4. 代码接口与失败语义

```python
SeedanceClient(config, transport=None).prepare_submit(payload)  # 无 key、无网络
SeedanceClient(config, transport=None).submit(payload)
SeedanceClient(config, transport=None).poll(task_id)            # 单次 GET，不等待
GeminiClient(config, transport=None).review(video_path, prompt, response_schema=None)
GeminiClient(config, transport=None).generate_image(image_path, prompt,
    model="gemini-nano-banana-2.1")
```

`config` 为字典，必填 `model`；可填 `base_url`、`api_key_env`、`timeout`。
默认超时 120 秒。`prepare_submit` 返回无鉴权信息的 `method/url/json`。
传入与配置不同的 model 被拒绝。provider 接受内容项时要求显式 role，便于审计。
返回公共字段 `status/task_id/output_url/usage/model`；Gemini 另返回 `review` 或
`images=[{"mime_type":...,"data":...}]`。

注入 transport 的协议为 `(method,url,*,headers,json,timeout)->dict`。
默认用标准库 urllib，禁止自动重定向，没有自动重试。
`ProviderError` 提供 `code/retryable/submission_unknown/http_status`：
POST 超时、传输中断、5xx 或无法确认任务 ID 时按 `submission_unknown` 处理，
不自动重发；GET 暂时失败可由外层稍后再查。
异常不包含原始 provider body、密钥、鉴权 header 或带签名的素材 URL。

离线验证：`PolicyGen/.venv/bin/python -m pytest -q data_generation_methods/tests/test_providers.py`。
35 项通过，覆盖请求格式、混用限制、超时、一次提交、匿名媒体、JSON 截断及锚帧返回；
它们验证协议实现，不声称真实账号权限、生成质量或付费链路已经验证。
