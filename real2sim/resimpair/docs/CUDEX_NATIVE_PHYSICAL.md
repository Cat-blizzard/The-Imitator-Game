# 实际 cuDex 原生 Panda 接入

先看[实测与完整边界](../validation/cudex_physical_20261026/RESULTS.md)。本轮完整抓放：几何 7/8，cuDex 优化候选 8/8；不是官方复现成功率，也不是全库估计。保留上一轮静力代理研究及其负证据，两个实验不能混称为同一方法。

代码：

- `resimpair_grasp/cudex_native.py`：原生 URDF 条件、实际 GPU 求解、原始结果与严格成功标记保存、显式候选导出、有限作业与独占 GPU 锁。
- `resimpair_grasp/cartesian_clearance.py`：按控制器实际世界 Z 方向检查整个原生夹爪的接近扫掠体积。
- `resimpair_pickplace/cartesian_reachability.py`：两组共用的选臂与源/目标端点检查。
- `resimpair_pickplace/cudex_benchmark.py`：相同初态的冻结八来源对照；不会更换失败来源。
- `resimpair_pickplace/packing.py`：1–6 个不同物体的保守底面容量分配、前序放置障碍与手部间隙检查；动态范围目前只测双物体。
- `resimpair_pickplace/cartesian_lift.py`、`cartesian_motion.py`、`cartesian_runner.py`：显式新控制候选。三次真实失败的 CPU 重规划通过，但新控制尚未做物理执行，未进入默认管线。

可选运行时使用独立依赖目录，不替换现有 `.venv`。安装前阅读 [上游许可](../third_party/cudex_native/LICENSE.upstream)，不是默认允许任意再分发的通用依赖。配置依据原生 Panda 几何适配，不能直接用于其他机器人。

```bash
# 此步骤仅安装/编译，不运行 GPU 优化或物理仿真。
.venv/bin/python scripts/setup_cudex_native.py --cuda-home /usr/local/cuda-12.9 --arch 12.0

# 只读已有批次的耗时/失败；不会追加调用。
.venv/bin/python -m resimpair_pickplace.batch_diagnostics \
  --campaign runs/cudex_physical_20261026/paired_physics \
  --campaign runs/cudex_physical_20261026/packing_physics \
  --output <new_report.json>
```

本轮优化 10/10、物理 20/20、API 0 已结束。`cudex_native.run_job` 当前绑定这轮有限协议，不能把换一个输出目录当作新额度；恢复以已有 reservation/process/terminal 为准。扩大优化前应把预算账本提取为跨输出目录的共同登记器，并登记明确新增额度；本轮没有通过换目录追加优化。

新控制候选要求独立 campaign，`controller: cartesian_lift_v1`，并绑定 `cartesian_runner.py`、`cartesian_motion.py`、`cartesian_lift.py` 的 SHA。它拒绝借用旧 controller campaign。此入口已可运行，但本轮物理预算已用完，不能直接对已有记录重复执行。

从真实失败保存的 `request.json`、状态、负载变换与网格调用 `cartesian_lift.replay`，无需加载仿真器和 GPU。离线守卫通过仅表示该条规划路径通过保留检查，不证明动态夹持或整段任务成功。

旧 `reachability.py`、`stability_motion.py`、物理验收器和 main 分支保留。没有真实来源窗口的组合、条件合同继续标记 `synthetic_only`。
