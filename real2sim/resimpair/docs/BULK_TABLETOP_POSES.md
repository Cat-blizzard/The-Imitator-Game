# 批量桌面姿态与抓取库

本轮目标是先从大库批量计算可用资产和桌面抓取 pose，再用小样本检验执行效果。**入库不要求每个资产跑一次动力学仿真。** 旧的 258 个加工资产、19 个操作几何候选是上一轮受限抽样和加工流程的结果，不是整个原始库的抓取能力上限。

## 当前方法

`resimpair_tabletop/` 以原始索引为输入，跳过大型家具、缺失/不支持的文件和超过本批资源预算的模型。没有“必须属于已手工支持的少数类别”的操作资格白名单；缺少语义信息的物体保留 `source_metadata_unverified`，仍可做几何计算。

1. 按内容哈希去重和划分 development / heldout_asset，同一网格的派生姿态不得跨划分。
2. 使用定向包围盒测量尺寸，跳过过宽扁/过细长的模型。统一缩放在搜索抓取之前确定：中间边目标 55 mm、最长边限定 65–160 mm。它是合成道具的工程尺寸，不是恢复了真实物体尺寸。
3. 计算最多三个重力支撑姿态；在完整视觉网格上批量射线搜索水平对向接触，检查夹爪开口、摩擦锥及离桌高度。
4. 使用 Panda 原生 URDF 的手掌、手指及腕部相机碰撞形状，检查全开夹爪和连续竖直接近扫掠体，防止只检查指尖却漏掉手掌/桌面/障碍。碰撞包含性另查，避免只有表面相交检测造成漏检。
5. 保存自包含纹理 GLB、缩放与支撑变换、各姿态的 TCP/contact/width、失败原因和计算时间。无需运行机械臂或动力学。
6. 任务要用某些物体时，才生成相应仿真碰撞模型并导出旧格式 `catalog.json` 和 `cache_index.json`。继续使用现有严格路径规划、动作/状态/双视角记录与独立轨迹验收。

`offline_tabletop_feasible` 表示在当前有限搜索与几何规则下有候选，不表示已验证力学稳定、全臂可达或所有任务都可完成。质量评估依靠预先冻结的随机样本；正式生成的每条轨迹仍经过原独立验收。

## 运行

原环境不变；快速射线库装在独立目录：

```bash
bash scripts/setup_tabletop.sh
bash scripts/run.sh tabletop plan --root runs/my_tabletop_library \
  --index assets/expanded_library/raw_roles_v4.sqlite \
  --profile configs/tabletop_bulk_v2.json --limit 12000 \
  --workers 24 --timeout 60 --disk-gib 64 --wall-seconds 7200 --seed 26102610
bash scripts/run.sh tabletop run --root runs/my_tabletop_library --limit 12000
bash scripts/run.sh tabletop report --root runs/my_tabletop_library
```

每个资产最多 60 秒、本批最多 24 个 CPU worker、64 GiB 新输出，保留至少 50 GiB 磁盘余量。进程锁和 reservation 防止重复运行；有未结束记录时先恢复核对。没有自动重抽，没有 API 调用，不使用训练 GPU。全量源数据没有复制进 Git；继续加工未转换资产需要原始源库，已经转换的 GLB 可独立使用。

扩大下一批时使用增量计划；它按内容身份跳过上一批的所有已完成尝试，失败也不静默重跑：

```bash
.venv/bin/python -m resimpair_tabletop.extend \
  --previous runs/my_tabletop_library/reports/REPORT_HASH.json \
  --root runs/my_tabletop_library_next --limit 12000
bash scripts/run.sh tabletop run --root runs/my_tabletop_library_next --limit 12000
```

这个入口递归继承旧批次，避免扩大总量时重算已有缓存。修改算法后重试失败应另设对照实验并保留旧失败，不能混入“首次处理成功率”。

查询已有 pose 并接到现有任务工厂：

```bash
.venv/bin/python -m resimpair_tabletop.query \
  --catalog runs/my_tabletop_library/reports/REPORT_HASH.json \
  --output runs/my_asset_selection --count 20 --partition development \
  --max-dimension 0.18 --seed 42
.venv/bin/python -m resimpair_tabletop.materialize run \
  --root runs/my_asset_selection --workers 2 --timeout 180
.venv/bin/python -m resimpair_tabletop.native_mass \
  --index runs/my_asset_selection/cache_index.json --output runs/my_asset_selection_mass
bash scripts/run.sh tasks init --root runs/my_tabletop_tasks \
  --catalog assets/expanded_library/scene_catalog_v3.json \
  --cache-index runs/my_asset_selection_mass/cache_index.json \
  --scenes 2 --families select clear arrange ordered --physical-limit 12 --gpu 2
```

查询可加 `--category`。查询只过滤来源类别和尺寸，不把模型文件名当成已确认语义。任务工厂仍需两个可用目标资产；若碰撞转换不足两个会报错，不能静默补入容易成功的资产。`tasks init` 不执行；`tasks run` 才消耗记录在案的物理执行预算。

`native_mass` 使用原生碰撞组件计算质量，在执行前校准密度到已声明的工程质量；不推进动力学、不读取执行结果、不改几何或旧验收规则。它修复多块碰撞体的网格体积估计与 PhysX 质量积分有小幅差异时，描述值和实测值不一致的问题。输出是独立版本，原描述、缓存和失败仍保留。

## 抽样和质量边界

主要验证协议见 `configs/tabletop_heldout_sample_v1.json`：完整候选库中留出的资产随机抽 24 个，转换失败也保留在分母，不补抽。分别统计：离线 pose 数量、碰撞转换成功数、实际执行数、抓取抬升通过数、完整 pick/place 通过数。小样本比例附置信区间，只代表登记的候选子总体。

原碰撞模型标准保持完整表面 3 mm 和原生重载 10 µm 校验。另有仅用于开发诊断的 `proxy.py` / `proxy_v2.py`：外包碰撞代理必须覆盖视觉网格、抓取两侧接触斑块误差合格，并重新通过全开夹爪接近检查；禁止用于插入、容器内腔、倒料等接触机制。它不声称通过旧的完整表面标准，也没有替换默认资产资格。v1 失败、v2 修正 PhysX 向内简化的记录分别保留。

几何统计不是“几千个已经仿真成功”的结论。未抽中的物体保持 `unmeasured`；碰撞转换、机器人可达、摩擦/质量近似、细小接触和放置稳定性仍可能导致实际失败。视觉资产更丰富也不会自动增加接触机制种类。

## 方法依据

[GraspVLA 原论文](https://arxiv.org/html/2505.03233v3)使用大规模物体与抓取合成，轨迹再进行动力学验证；它不意味着任意几何 pose 都保证真实抓取成功。本实现复用本仓库现有严格执行器，并采用 [trimesh 的 Embree 射线接口](https://trimesh.org/trimesh.ray.ray_pyembree.html)加速离线接触搜索，未复刻或声称重现论文系统。
