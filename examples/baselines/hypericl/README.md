# HyperICL-H2R on The Imitator Game — 两个版本的实现

目录放到 `examples/baselines/hypericl/`，与 `examples/baselines/template/` 同级。两个脚本的接口与
`train_template_imitator.py / eval_template_imitator.py` 完全一致（`compute_loss / prepare_for_eval /
get_action / clear_cache`，checkpoint 含 `args` + `agent_state_dict`），评测直接复用 Diffusion Policy 的仿真流程。

设计文档见 `../DESIGN_H2R_HYPERNET.md`（架构、监督项、训练流程的完整说明）；本文只讲怎么跑、两版差异、参数量对齐、以及你需要接的几个数据钩子。

```text
hypericl/
├── config.py                      HyperICLArgs（tyro CLI，template TrainingArgs 的超集）
├── agent.py                       HyperICLAgent：两版共用（E / T / D / TTT / 专家库 / 全部 loss / 闭环递推）
├── expert_bank.py                 Stage B：按任务索引的 LoRA 专家库（参数教师、函数教师、字典初始化）
├── train_hypericl_imitator.py     分阶段训练 A→E；打印可训练参数分解
├── eval_hypericl_imitator.py      仿真评测；--ttt-steps / --state-update-every 切换 one-shot / TTT / 闭环
├── targets/
│   ├── base.py                    TargetPolicy 接口
│   ├── scratch_target.py          版本 1：FM-DiT from scratch
│   └── pretrained_target.py       版本 2：冻结 VLA/WAM + 注入生成 LoRA（lerobot SmolVLA / π0；GR00T 只给 regex）
└── modules/
    ├── frame_tokenizer.py         冻结逐帧 token（benchmark backbone 钩子 或 DINOv2 hub）
    ├── video_program_encoder.py   E：effect / event / progress / embodiment + 未来预测 + 域分解
    ├── hyper_compiler.py          T（递推 skill state）、D（字典系数 + 零初始化残差 LoRA + 门 + 阶段偏置）、P_phi（latent TTT）
    ├── lora_dit.py                LoRALinear（每样本 delta）、LoRADiT、lora_context
    ├── world_model.py             latent 动力学 W + effect head
    ├── heads.py                   flow matching
    └── losses.py                  TCC / soft-DTW / 顺序 / 参数教师 / 流式参数统计 / 正则
```

---

## 1. 两个版本

| | 版本 1 `--target scratch` | 版本 2 `--target vla` |
| --- | --- | --- |
| 目标策略 θ0 | `LoRADiT`（dim 256 / depth 6 / FM，Stage A 从零训练，之后冻结） | 预训练 VLA 的 action expert（SmolVLA / π0 的 flow-matching expert），Stage A 训一套 task-agnostic **domain LoRA** 后合并进权重 → θ0，之后冻结 |
| 策略的感知 | benchmark 的 `simple_cnn` + state MLP（Stage A 训练） | VLA 自带的 SigLIP/VLM prefix（永远冻结） |
| 生成参数落在哪 | 每个 DiT block 的 q/k/v/o/fc1/fc2（r=8）、ada（r=16 + 完整/阶段偏置）、输入/输出层 | expert 每层的 q_proj / v_proj / o_proj / down_proj（r=8）；由 `--vla-lora-regex` 控制 |
| 一个 skill token 对应 | 一个 DiT block（L = depth + 2） | 一个 expert layer（L = 匹配到的层数，可用 `--vla-max-layers` 只取最后 n 层） |
| 共用部分 | E、T、D、P_phi、专家库、W、全部 loss、闭环递推、评测脚本 | 同左，一行不改 |
| hypernet 侧的机器人 latent h_t | 同一套 `simple_cnn`+MLP（也作为 DiT 的条件） | 同一套 `simple_cnn`+MLP（只喂 hypernet，Stage C/D 可训练） |

为什么 VLA 版这么接：WIZARD (2606.07217) 给冻结 π0.5 生成 LoRA、RoboTTT (2607.15275) 把 fast weights 放进 GR00T 的 DiT action head——都只动 action expert、不动 VLM。`PretrainedTarget` 是通用注入：按正则把 `nn.Linear` 原地换成 `LoRALinear.from_linear`，deltas 通过 `lora_context(P)` 传入（不改 VLA 的 forward 签名），速度场用 forward hook 从 `action_out_proj` 抓出来做函数空间损失。

## 2. 训练命令（两版一样，只换 `--target`）

```bash
# 版本 1：from scratch
T=scratch; P=examples.baselines.hypericl.train_hypericl_imitator
python -m $P --target $T --stage A --exp-name ${T}_A --total-epochs 20
python -m $P --target $T --stage B --exp-name ${T}_B --init-from runs/${T}_A/checkpoints/final_model.pt --total-epochs 10
python -m $P --target $T --stage C --exp-name ${T}_C --init-from runs/${T}_B/checkpoints/final_model.pt --total-epochs 10
python -m $P --target $T --stage D --exp-name ${T}_D --init-from runs/${T}_C/checkpoints/final_model.pt --total-epochs 20
python -m $P --target $T --stage E --exp-name ${T}_E --init-from runs/${T}_D/checkpoints/final_model.pt \
       --total-epochs 10 --ttt-steps-train 1 --p-img 0.3

# 版本 2：预训练 VLA（SmolVLA 示例；π0 把 --vla-profile 换成 lerobot_pi0 并给 --vla-repo-id）
T=vla
python -m $P --target $T --vla-profile lerobot_smolvla --vla-repo-id lerobot/smolvla_base --stage A --exp-name ${T}_A --batch-size 64
# 之后 B/C/D/E 同上（每个命令都带 --target vla --vla-profile ... 以便重建；batch 视显存调小）
```

每次启动都会打印 `trainable parameters (M)` 的逐模块分解（E / T / D / ttt / fb / wm / effect / bank / target），
用它做两版对齐，而不是估算。

评测：

```bash
python -m examples.baselines.hypericl.eval_hypericl_imitator --eval-config <exp_list.txt> \
  --checkpoint runs/${T}_E/checkpoints/final_model.pt --output-dir results/${T}_E \
  --human-config ... --sim-config ... --task-mapping ... --human-task-desc ... --sim-task-desc ... \
  --ttt-steps 0 --state-update-every 0      # one-shot
  --ttt-steps 2 --state-update-every 0      # + latent TTT
  --ttt-steps 2 --state-update-every 1      # + 闭环递推（论文主设置）
```

## 3. 可训练参数量对齐

默认配置下的解析估计（DiT-256/6；SmolVLA expert 取 hidden≈720、16 层、每层 4 个线性层；精确数看启动打印）：

| 模块 | scratch | vla |
| --- | --- | --- |
| E（视频程序编码器，含未来预测头） | ≈5.3M | ≈5.3M |
| T（ContextTransformer，4 层） | ≈3.2M | ≈3.2M |
| D（共享残差头 + 系数头 + 门 + 阶段偏置头） | ≈5–6M | ≈2.5M |
| 流形原子（住在 target 的 LoRALinear 里，Stage D/E 可训练） | ≈6.7M | ≈14M（M=16）/ ≈7M（M=8） |
| P_phi、fb、W、effect、prog head | ≈2.5M | ≈2.5M |
| obs 编码器 + obs_proj | ≈2–3M（Stage A） | ≈2–3M（Stage C/D，只喂 hypernet） |
| θ0 | DiT ≈7.4M（Stage A 训练，之后冻结） | domain LoRA ≈1M（Stage A，合并后冻结）；VLA 450M 冻结 |
| **Stage D/E 可训练合计** | **≈24M** | **≈30M（M=16）→ ≈23M（`--n-atoms 8`）** |

对齐旋钮：scratch 侧 `--dit-dim / --dit-depth / --n-atoms`；vla 侧 `--n-atoms / --rank / --vla-max-layers / --vla-lora-regex`。
论文里同时报"可训练参数量"和"每任务生成参数量"（scratch ≈0.45M/任务；vla 默认 ≈0.9M/任务）。

## 4. 你需要接的数据钩子（都是可选，缺了会自动关掉对应 loss）

| 需要 | 在哪用 | 缺了会怎样 |
| --- | --- | --- |
| `human_video` 的**逐帧 patch token** `[B,N,P,F]` | E 的全部时序结构 | `--tokenizer-backend template` 时如果 backbone 只返回 `z`，自动退化成单 token（= 旧的 mean-pool），并打印一次警告。在 `video_backbone.py` 里把 pooling 之前的 token 以 `out["tokens"]` 返回即可；或者用 `--tokenizer-backend dinov2_hub`（自带 DINOv2，`--feat-dim 1024`） |
| `robot_progress ∈ [0,1]` 或 `robot_frame_index` + `robot_episode_length` | L_prog 的机器人侧监督、feedback token 的 chunk 编号 | 只剩 human 侧 TCC/顺序损失；chunk 编号取 0 |
| `robot_next_obs`（t+pred_horizon 的观测，格式同 `robot_obs`） | L_world | W 只由 L_effect 训练 |
| `robot_video`（机器人相机流的 N 帧） | 域分解损失 + human↔robot 进度对齐（TCC + soft-DTW） | 域损失关闭 |

在 `lerobot_paired_dataset.py` 的 `__getitem__` 里各加一个 key 即可，collate 已按 key 透传。

## 5. VLA 版本上线前的核对清单（我这边没有 lerobot 环境，属性名按公开源码写，**必须**跑一次 Stage A 的 1 个 step 确认）

1. `PretrainedTarget._inject` 会打印注入了多少个 `LoRALinear`；如果正则没匹配到，会列出前 40 个 Linear 名字给你改 `--vla-lora-regex`。
2. `policy.forward(batch, noise=..., time=...)` 接受 `noise/time` 时函数空间损失（L_func / L_intra）和 L_effect 才可用；不接受时会自动退回 `forward(batch)`，这两项关闭。
3. 速度场 hook 默认挂在 `model.action_out_proj`；名字不对会打印提示并关闭函数空间损失。
4. 采样优先 `predict_action_chunk`，其次 `_get_action_chunk`，最后 `select_action × horizon`（依赖 `n_action_steps == pred_horizon`）。
5. Imitator 数据已经 `bounds_q99` 归一化：默认 `--vla-identity-norm` 把 `normalize_inputs / normalize_targets / unnormalize_outputs` 换成 Identity；若你的 lerobot 版本用 pre/post processor，请把它们配成 identity。
6. 图像 key 顺序来自 `policy.config.image_features`；单相机默认取第一个 key。语言 prompt 统一为 `--vla-task-text`。
7. 动作 chunk：VLA 的 `chunk_size`（SmolVLA 50）> `pred_horizon`（16）时按最后一个动作补齐并给 `action_is_pad`，loss 只算前 16 步。

## 6. 尚未实现 / 已知限制

- 我这边的沙箱没有 torch，代码只过了语法检查和逐张量形状人工核对；第一次跑请用 `--batch-size 8 --total-epochs 1` 走通 A→E。
- 多 chunk 的 truncated BPTT（RoboTTT 式）需要数据集提供同一 episode 的连续 chunk；当前 Stage E 用"当前观测反馈 token 的第二遍前向 + 想象反馈"近似（`--feedback-pass --p-img`）。数据集能给 `episode_index/frame_index` 后，在 `compute_loss` 里把 `S0[inv] → S1` 的一步递推扩成 C 步即可。
- Stage F（只对 skill token 做 RL）没写：需要仿真闭环 rollout 的接口，设计见 DESIGN §6。
- `soft_dtw` 是 O(NM) 的 Python 双循环（N=M=10 帧无所谓；帧数上 64 再换 CUDA 版）。
- GR00T N1.5 只给了 regex profile，`_load` 需要你接 Isaac-GR00T 的 policy 对象（`PretrainedTarget.from_policy`）。
