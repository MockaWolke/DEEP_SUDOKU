cd /home/felix/Desktop/DEEP_SUDOKU/reinforcement_learning_experiment

python multi_action_ppo_script.py \
    --exp-name MultiActionSeperateOnlyConv  \
    --total-timesteps 3000000 \
    --gym-id Sudoku-nostop0 \
    --mask-actions True \
    --agent MultiActionSeperateOnlyConv \
    --eval-freq 50 \
    --eval-steps 100 \
    --anneal-lr True \
    --upper-bound-missing-digits 5 