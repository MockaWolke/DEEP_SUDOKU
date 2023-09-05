cd /home/felix/Desktop/DEEP_SUDOKU/reinforcement_learning_experiment

python single_action_ppo_script.py \
    --total-timesteps 10000000 \
    --gym-id Sudoku-x2 \
    --exp-name measure_time_x2 \
    --mask-actions True \
    --agent SeperateOnlyConv \
    --anneal-lr False \
    --eval-freq 50 \
    --eval-steps 100 \