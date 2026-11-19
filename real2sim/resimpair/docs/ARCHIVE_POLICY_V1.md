# 可选的无损分割归档策略

新入口为 `resimpair_tasks.archive_policy_adapter_v1`，适用于已准备的
`closure_background_replay_v1` 原生状态回放任务。旧渲染器和旧入口均未修改；
**默认仍为原来的 DEFLATE6，只有明确选择 `deflate1` 才改变压缩级别。**

它只替换 worker 内 `scene_quality_render_v3` 模块引用的 NumPy 视图，将该
worker 的 `segmentation.npz` 保存调用接到显式 writer。普通 NumPy 操作仍委托
原模块；全局 `numpy.savez_compressed` 不变；正常返回或异常时都恢复模块引用。
文件成员、数组顺序、形状、dtype 和像素值不变，不减少帧或删除可见性证据。

在一个**尚未提交过任何作业**的新背景批次已经完成原来的准备、预算登记后，
运行下面命令。`NEW_PREPARED_ROOT` 应换成该新批次路径，不要指向已经完成的
`runs/contact_closure_background_20261030_v1`。

```bash
.venv/bin/python -m resimpair_tasks.archive_policy_adapter_v1 prepare-policy \
  --root NEW_PREPARED_ROOT --archive-policy deflate1

.venv/bin/python -m resimpair_tasks.archive_policy_adapter_v1 run \
  --root NEW_PREPARED_ROOT --stage screen --limit 2

# 完成独立 screen 审核，取得该批次的真实 gate 后再执行：
.venv/bin/python -m resimpair_tasks.archive_policy_adapter_v1 run \
  --root NEW_PREPARED_ROOT --stage video --limit 2 --gate INDEPENDENT_SCREEN_GATE.json
```

省略 `--archive-policy` 会冻结 `deflate6`。策略写入不可覆盖的
`archive_policy_v1.json`，绑定原 protocol、预算、worker、writer 和适配器版本；
批次已有任何 reservation 就拒绝更换策略。`run` 继续使用原 supervisor 的
GPU 锁、预算、超时、状态未知恢复和独立视频门禁。没有新增或重置预算。

每个 worker 另存 `archive_policy_entry.json`、`archive_policy_result.json`，
记录实际调用、保存耗时、数组形状/dtype、输出哈希和原 execution bundle 的绑定。
这些侧车必须与原 bundle 一起保留；不能把新策略来源伪装成只有旧渲染器的执行。

实际证据分为三层：

- **原完整视频耗时**：TinCup 393 帧、Medikit 1,000 帧，各含两个视角，原方式总耗时
  85.80 秒和 173.38 秒。已经完成的四次 screen/video 作业保持原数据和原压缩方式。
- **64 帧无损对照**：固定真实 TinCup 两视角的 64 个均匀抽取帧，串行一次测量。
  DEFLATE6 写入 4.362 秒、1.985 MB；DEFLATE1 写入 2.366 秒、4.339 MB。
  解码分别 0.880 秒、1.096 秒。数组逐项相等，解压后的完整 NPY 成员 SHA 相同。
  这是写入时间与存储大小的取舍，不能推断整条视频按 1.84 倍加速。
- **worker 保存接口验证**：从冻结的真实 worker 源码提取原保存语句，经新适配器
  对同一真实 64 帧执行两种策略。数组与 NPY 成员再次完全一致；检查了全局 NumPy
  不变、引用恢复、默认选择和拒绝覆盖。没有运行新的仿真或 GPU 渲染。

证据目录：`runs/segmentation_archive_benchmark_20261031_v1/`；其中
`RESULTS.json` 是独立 writer 微基准，`worker_interface/RESULTS.json` 是保存调用
接入验证。`tests/test_archive_policy_adapter_v1.py` 验证软件边界。

**尚未进行新策略下完整视频的端到端性能对照，未晋级默认管线。**
