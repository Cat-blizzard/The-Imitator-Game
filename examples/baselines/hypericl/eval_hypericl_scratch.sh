T=scratch
python -m examples.baselines.hypericl.eval_hypericl_imitator \
  --checkpoint runs/${T}_D/checkpoints/final_model.pt --output-dir results/${T}_D \
  --eval-config examples/baselines/lerobot_dataset/eval/exp_list/seen_5tasks_env_list.txt \
  --human-root /mnt/sdf/engram_sz_DataCollection/imitator_zed2i_224 --sim-root /mnt/sdf/engram_sz_DataCollection/IG-10K-Dataset/imitator_sim_v1_zed2i \
  --human-config examples/baselines/lerobot_dataset/config/exp_configs/human_test_config_seen.json --sim-config examples/baselines/lerobot_dataset/config/exp_configs/sim_test_config_seen.json \
  --task-mapping examples/baselines/lerobot_dataset/task_mapping.json \
  --human-task-desc examples/baselines/lerobot_dataset/task_desc/human_desc.json --sim-task-desc examples/baselines/lerobot_dataset/task_desc/sim_desc.json \
  --ttt-steps 0 --state-update-every 0