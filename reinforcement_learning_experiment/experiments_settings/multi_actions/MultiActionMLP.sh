cd /home/felix/Desktop/DEEP_SUDOKU/reinforcement_learning_experiment

python multi_action_ppo_script.py \
    --exp-name MultiActionMLP \
    --total-timesteps 5000000 \
    --gym-id Sudoku-x2 \
    --mask-actions True \
    --agent MultiActionMLP\
    --eval-freq 50 \
    --eval-steps 100 \
    --anneal-lr True \
    --upper-bound-missing-digits 5 