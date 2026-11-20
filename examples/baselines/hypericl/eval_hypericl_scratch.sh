T=scratch
python -m examples.baselines.hypericl.eval_hypericl_imitator \
  --checkpoint runs/${T}_E_refine/checkpoints/final_model.pt --output-dir results/${T}_E_refine_unseen \
  --eval-config imitator/eval/lists/exp_list/unseen_5tasks_env_list.txt \
  --human-root /mnt/sdf/engram_sz_DataCollection/imitator_zed2i_224 --sim-root /mnt/sdf/engram_sz_DataCollection/IG-10K-Dataset/imitator_sim_v1_zed2i \
  --human-config imitator/dataset/config/exp_configs/human_test_config_unseen.json --sim-config imitator/dataset/config/exp_configs/sim_test_config_unseen.json \
  --task-mapping imitator/dataset/task_mapping.json \
  --human-task-desc imitator/dataset/task_desc/human_desc.json --sim-task-desc imitator/dataset/task_desc/sim_desc.json \
  --ttt-steps 0 --state-update-every 0