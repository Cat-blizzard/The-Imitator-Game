# 同步多视角候选记录器（2026-10-21）

入口为 `strict_apple_adapter.py --multiview`，实现为
`multiview_recording.py`。它保留 `zed2i` 224 和 `native` 512 的原位置，
并在同一环境增加 `opposite` 512。原 `rich_runtime_v2.py`、物理规划、
物理验收和原视角质量门均未修改；不带参数仍使用原记录器。

目前状态：已接入实际 strict runner；9 项 CPU 记录器测试通过，
其中编码/解码使用实际 H264 与 PyAV。CPU 测试中的环境是明确的测试
fixture，不是物理资格证据。实际仿真、独立证据审计及相机收益由本轮
不可变运行目录的 `result.json` 和独立审计报告确定，不能从本文件推定。

实现选择来自既有失败：旧 `opposite_low` 是替换主视角并重新执行。
本候选增加相机时不增加 `env.step`、不移动原相机，也不重新执行动作。
新增相机使用原诊断的低对面机位（eye=[0.45,0,0.45]，
target=[0,0,0.12]，FOV=1.3），尚未在更多任务上资格。

每步从同一个 `obs` 读取三路 RGB 和 QA segmentation，在原
`Recorder.step` 所记录的状态与命令之前捕获。原两路 RGB 沿用原
recorder 存储，新增对面帧保存独立副本。每帧检查原两路 RGB hash
与原 recorder 实际保存内容一致；相机内外参发生变化、缺相机、
非有限命令会停止。若 `env.step` 在记录后抛错，仍保留该次
pre-command 记录，失败产物不会获得成功标签。

正常导出保留原文件，新增：

- `episode/video_opposite512.mp4`：与原两路同 FPS、同长度。
- `episode/multiview/timeline.npz`：三路帧到共享 state/action 索引，
  相对时间、环境累计控制步数、原始 RGB hash、对象像素数及可见 bbox。
- `episode/multiview/manifest.json`：执行 UUID、所有视频与共享
  `arrays.npz` 的 SHA256、相机分辨率、内外参、近远裁面、FOV，
  对象身份和动态轨迹去重 hash。
- receipt 以 hash 绑定上述 manifest；input 与 source_archive
  冻结执行 ID、候选配置和新增源文件，在创建环境之前完成。

失败时保留原 `failure/trace.npz`，额外导出
`multiview_shared_arrays.npz`、三路视频及
`diagnostic_failure_trace=true` 的 manifest。这表示尝试过的命令和
命令前观察，不保证抛错命令实际完成。

动态轨迹 hash 与 20261020E 的字段顺序一致：
`native_commands + states + object_states + timestamps_seconds` 的
C-contiguous 原始字节。一个执行只有一份共享轨迹，不把三路视频
记为三条动作数据。两个不同机位、三个传感器流、独立执行次数和
不同动态轨迹数必须分别统计。

额外开销为一台 512×512 相机的 RGB/segmentation 渲染、原始 RGB
内存（每帧 786,432 字节）、逐帧相机参数读取/哈希/分割统计，
及一次 H264 编码。没有新增 API。实际运行时间和产物大小须实测。

可见性来自仿真对象 segmentation；它只是参考视角筛选证据，
不是合成人类接触、遮挡语义或真实 3D 接触真值。固定机位变化会
被拒绝；此版本不支持移动相机，也不通过改阈值补救原视角质量失败。

```bash
PolicyGen/.venv/bin/python -m unittest \
  data_generation_methods.tests.test_multiview_recording \
  data_generation_methods.tests.test_strict_motion -v
```

候选执行应由主执行器检查 GPU、锁、进程和剩余磁盘后安排一个
独立输出目录。未通过新的独立审计前，不晋级为默认生产记录器。
