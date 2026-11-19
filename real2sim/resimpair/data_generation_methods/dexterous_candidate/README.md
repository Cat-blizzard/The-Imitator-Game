# 独立机械臂＋灵巧手候选（2026-10-21）

本候选复用本地 Panda 与 Allegro 右手资产，通过声明的固定适配器组合成 7＋16＝23 自由度本体，并实际执行接近、下降、闭手、抬手、保持和释放。接入与控制已完成；本轮没有成功抓取。旧生产代码与原始实验未修改。

**资格边界更新：这是控制接口候选，动力学尚未资格。** 更严格的源惯性检查发现 Allegro 的 13 个 link 违反主惯量三角不等式，最大绝对违反量 `1.5804e-5 kg·m²`，最大相对最大主惯量违反量 `28.23%`，不能解释为微小舍入。源惯性保持不变；PhysX 对这些源张量的实际加载处理未导出，因此模拟接触数值不能用作硬件可信度证据。`evidence_summary/inertia_triangle_audit.json` 保留量化结果。下一轮应优先核对/修复资产，而不是调抓取参数制造成功。

结果根目录：`/mnt/sdg/datasets/real_sim_human_pipeline/report_control_20261021/dexterous/`。

- `assets.py`：静态资产清点、依赖哈希和 URDF 组合。保留源惯性；固定适配器平移 `[0,0,0.14] m`、零旋转，没有额外质量/CAD，不能视为真实硬件资格。
- `runtime.py`：首轮实际控制入口，原始失败保留在 `grasp_attempt_01/`。其接触分组漏掉固定指尖，原始 0 N 不能表示没有接触。
- `runtime_contact_v2.py`：只复放保存的 300 条命令，包含全部四指固定指尖，240 Hz 保存每个接触段的力。机器人、材质、相机、控制器、初态、种子及成功条件不变。`contact_replay_02/` 与首轮动作及状态逐元素相同。
- `analyze.py`：CPU 汇总、模型惯性/树结构检查、真实力/物体位移图、首行控制占位语义与有效行掩码。
- `runtime_backend_v3.py`：显式选择 `PolicyGen` 或 `The-Imitator-Game`，拒绝混用已导入的 ManiSkill，记录实际 `mani_skill`、`BaseEnv`、`BaseAgent` 源文件哈希。`imitator_game_replay_03/` 已在 The-Imitator-Game 后端实际复放同一 300 命令，状态逐元素一致。Python 环境和绑定资产仍复用 PolicyGen，不能称为资产自足移植。

The-Imitator-Game 内的独立入口是 `The-Imitator-Game/examples/dexterous_candidate/replay_panda_allegro.py`；只接受输出、URDF 和源动作文件三个参数，固定选择该仓库后端。原有任务/默认 agent 注册没有被改写。

冻结验收同时要求物体上升 5 cm、拇指与至少另一指接触、速度低于 0.05 m/s，并持续保持 20 个 20 Hz 控制帧。本次方块仅被中指尖推动约 6.12 cm，峰值 7.71 N；拇指接触为零，物体最大升高 0.076 mm。不能使用已有 UR10e-Shadow `is_grasping()` 的无条件 `True` 作为证据。

复放命令（需要先确认物理 GPU2 空闲，并与其他实验协调窗口；程序还会自行检查 UUID 和共享锁）：

```bash
PolicyGen/.venv/bin/python -m data_generation_methods.dexterous_candidate.runtime_contact_v2 \
  --output /mnt/sdg/datasets/real_sim_human_pipeline/report_control_20261021/dexterous/NEW_REPLAY \
  --urdf /mnt/sdg/datasets/real_sim_human_pipeline/report_control_20261021/dexterous/panda_allegro_model/panda_allegro_candidate.urdf \
  --replay-actions /mnt/sdg/datasets/real_sim_human_pipeline/report_control_20261021/dexterous/grasp_attempt_01/trajectory.npz
```

原始 NPZ 有 301 个状态、300 个实际控制命令。`actions[0]` 是未执行的初始化占位，采用 articulation active-joint 顺序；`actions[1:]` 才是 controller-joint 顺序的实际命令。两套顺序均保存于 `runtime_model.json`。应使用 `evidence_summary/valid_action_mask.npy`，不能把首行作为真实命令。复放入口跳过首行。原始数据不就地更正。

两个视频画面来自同一次执行的 overview/close 相机，不能计作两条动作。测量复放是第二次执行，The-Imitator-Game 兼容复放是第三次执行，合计仍是一套动作计划。API 调用 0；GPU 仅物理 GPU2，CPU PhysX、GPU2 渲染，使用共享 `gpu2.lock`。没有策略训练。

未完成：独立全臂路径碰撞资格、真实适配器建模、多指闭合协同和稳定抓取。下一步应先用手掌坐标系与接触几何定义物体位置和拇指对置策略，以接触反馈控制各指；不应只随机改闭合角度继续尝试。
