T=scratch; P=examples.baselines.hypericl.train_hypericl_imitator
python -m $P --target $T --stage A --exp-name ${T}_A --total-epochs 20
python -m $P --target $T --stage B --exp-name ${T}_B --init-from runs/${T}_A/checkpoints/final_model.pt --total-epochs 10
python -m $P --target $T --stage C --exp-name ${T}_C --init-from runs/${T}_B/checkpoints/final_model.pt --total-epochs 10
python -m $P --target $T --stage D --exp-name ${T}_D --init-from runs/${T}_C/checkpoints/final_model.pt --total-epochs 20
python -m $P --target $T --stage E --exp-name ${T}_E --init-from runs/${T}_D/checkpoints/final_model.pt \
       --total-epochs 10 --ttt-steps-train 1 --p-img 0.3