# 家居资产扩充：下载与索引

本轮实际数据盘：`/mnt/sdg/datasets/open_assets/household_scaleup_20261020_v1/`。
2026-10-20 迁移时 `/mnt/sdg` 约有 7.2 TiB 可用，`/mnt/sdf` 约 275 GiB。
代码仍保存在本目录；模型、贴图、压缩包、下载日志与清单均写入 SDG。

## 当前来源

| 子目录 | 来源与用途 | 本轮空间上限 | 许可处理 |
|---|---|---:|---|
| `objaverse/` | 筛选家居、厨房、收纳、工具等原始 GLB | 550 GiB | 逐对象保留 CC0 / CC-BY / CC-BY-SA 等明确许可；未确认不入成功库 |
| `abo/` | Amazon Berkeley Objects 家居产品、家具与日用品 | 200 GiB | 官方 CC-BY-4.0；保留产品、几何元数据与原始 GLB |
| `scanned_objects/` | Google Scanned Objects；YCB 实物扫描 | 100 GiB | 本轮 GSO 网格只下载原 Fuel CC-BY-4.0；Kubric 仅分类元数据 CC-BY-SA-4.0，未下载派生网格；已有 YCB 只引用 |
| `environments/robocasa/` | 厨房物体、家具组件、场景纹理 | 90 GiB | 官方资产 CC-BY-4.0；保留目录依赖和原始压缩包 |
| `environments/polyhaven/` | 家居几何、材质、HDRI | 9.9 GiB | CC0；保留作者、来源；Powered by Poly Haven |

这些是容量上限，不是预计下载量或已完成数量。各下载器另保留磁盘至少 1 TiB 可用空间。
采用有限并发、断点续传；已经完成并通过校验的文件直接复用。
不需要模型训练 GPU。不修改服务器全局代理；需要网络代理的进程使用现有本地代理。

## 下载完成与可用于仿真是不同状态

1. 保存许可、下载 URL、原始 ID、版本、文件长度、SHA256；有发布方哈希时进行核对。
2. GLB 检查容器结构与外部依赖；扫描物体和 Poly Haven 检查网格/材质可读性；Objaverse/ABO 另做实际 CPU 抽检。
3. RoboCasa 保留 MJCF 及相对引用目录，检查依赖；一个 XML 定义或一个纹理文件不能算一件独立可操作物体。
4. 原始资源不做猜测缩放，不用凸包替换容器内部几何。后续按任务核验尺度、碰撞、关节、质量/摩擦、可抓取性，再进仿真。
5. 当前库统一 `physics_qualified=false`、`training_admitted=false`；下载通过并不等于已生成训练轨迹。

RoboCasa 的 1594 个 AIgen 原 XML 引用了作者机器上的绝对 `ceramic.png` 路径，
对应贴图实际已包含在官方 textures 包。本轮保留原 XML，并生成只修正资源引用的 portable XML；
不修改网格、关节或物理参数。统一索引分别提供 `primary_path` 与 `portable_primary_path`，
以及各自的可读性/缺失依赖检查。便携转换完成前相关字段可能尚未写入。

## 数量口径与查看进度

统一索引在数据根目录的 `index/catalog.jsonl`，摘要在 `index/STATUS.md`、`index/STATUS.json`。
它仅收录各来源的成功清单，候选元数据和未下载完成的 `.part` 不计成已下载模型。
按来源 ID 去重；已有 YCB、MJCF 定义、背景纹理、HDRI 分别统计。
跨来源几何重复还需进一步判定，不能将来源计数相加后称“全局去重物体数”。
RoboCasa 重命名后的部分物体没有可追溯的原 Objaverse UID，不能臆造映射。

刷新快照：

```bash
/mnt/sdf/dataset/zhouxunzhe/zhiqi/PolicyGen/.venv/bin/python \
  /mnt/sdf/dataset/zhouxunzhe/zhiqi/data_generation_methods/asset_library/build_index.py
```

有限持续刷新可增加 `--watch-seconds 120 --hours 24`。各来源自己的 `LAUNCH.json`、
`process.json`/`OWNER.json`、日志与 `state.json` 记录实际进程及下载结果。
若下载进程终止，先检查日志，按同一根目录重启对应脚本即可续传；不要把候选清单当成功清单。

## 迁移记录

从 SDF 迁移的 116 个文件共 12,897,965 字节，逐文件 SHA256 相同且源文件未变化后移除旧副本。
证据保存在新根目录 `MIGRATION_FROM_SDF.json`；旧父目录只保留 `.MOVED_TO_SDG.json` 指向记录。
历史来源凭据中的旧绝对路径保持原样，迁移映射记录在证据内；当前下载清单使用新 SDG 路径。
其他历史资产库没有迁移，按已有来源 ID 引用或跳过以减少重复下载。

## 官方来源

- Objaverse: https://objaverse.allenai.org/objaverse-1.0/
- ABO: https://registry.opendata.aws/amazon-berkeley-objects/
- GSO: https://fuel.gazebosim.org/1.0/GoogleResearch/models
- YCB: https://www.ycbbenchmarks.com/
- RoboCasa: https://huggingface.co/datasets/robocasa/robocasa-assets
- Poly Haven: https://polyhaven.com/

后续造任务仍执行“先确认人类视频任务窗口，再定义机器人等价任务、少量验证、批量随机化”的要求。
资产扩充本身不改变这条配对约束。
