cd /home/felix/Desktop/DEEP_SUDOKU/reinforcement_learning_experiment

python single_action_ppo_script.py \
    --total-timesteps 5000000 \
    --gym-id Sudoku-v0 \
    --exp-name Sparse_ENV \
    --mask-actions True \
    --agent SeperateOnlyConv \
    --eval-freq 50 \
    --eval-steps 100 \
    --anneal-lr True \
    --upper-bound-missing-digits 5 