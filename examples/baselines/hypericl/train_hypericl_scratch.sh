#!/usr/bin/env bash
set -euo pipefail

GPU="${GPU:-3}"
SUBSET="${SUBSET:-45}"
export CUDA_VISIBLE_DEVICES="$GPU"

T=scratch
P=examples.baselines.hypericl.train_hypericl_imitator
DATA_CONFIG_DIR=imitator/dataset/config/exp_configs
COMMON_ARGS=(
  --target "$T"
  --human-dataset-file "${DATA_CONFIG_DIR}/human_train_config_${SUBSET}.json"
  --sim-dataset-file "${DATA_CONFIG_DIR}/sim_train_config_${SUBSET}.json"
)

python -m "$P" "${COMMON_ARGS[@]}" --stage A --exp-name "hypericl_${SUBSET}/${T}_A" --total-epochs 20
python -m "$P" "${COMMON_ARGS[@]}" --stage B --exp-name "hypericl_${SUBSET}/${T}_B" --init-from "runs/hypericl_${SUBSET}/${T}_A/checkpoints/final_model.pt" --total-epochs 10
python -m "$P" "${COMMON_ARGS[@]}" --stage C --exp-name "hypericl_${SUBSET}/${T}_C" --init-from "runs/hypericl_${SUBSET}/${T}_B/checkpoints/final_model.pt" --total-epochs 10
python -m "$P" "${COMMON_ARGS[@]}" --stage D --exp-name "hypericl_${SUBSET}/${T}_D" --init-from "runs/hypericl_${SUBSET}/${T}_C/checkpoints/final_model.pt" --total-epochs 20
python -m "$P" "${COMMON_ARGS[@]}" --stage E --exp-name "hypericl_${SUBSET}/${T}_E" --init-from "runs/hypericl_${SUBSET}/${T}_D/checkpoints/final_model.pt" \
  --total-epochs 10 --ttt-steps-train 1 --p-img 0.3
