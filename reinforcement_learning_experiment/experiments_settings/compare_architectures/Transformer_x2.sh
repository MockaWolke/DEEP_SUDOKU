cd /home/felix/Desktop/DEEP_SUDOKU/reinforcement_learning_experiment

python single_action_ppo_script.py \
    --total-timesteps 5000000 \
    --gym-id Sudoku-x2 \
    --exp-name Archi_TransformerAgent \
    --mask-actions True \
    --agent TransformerAgent \
    --eval-freq 50 \
    --eval-steps 100 \
    --anneal-lr True \
    --upper-bound-missing-digits 5 