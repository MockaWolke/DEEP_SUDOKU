cd /home/felix/Desktop/DEEP_SUDOKU/reinforcement_learning_experiment

python single_action_ppo_script.py \
    --total-timesteps 10000000 \
    --gym-id Sudoku-nostop0 \
    --exp-name measure_time_nostop \
    --mask-actions True \
    --agent SeperateOnlyConv \
    --eval-freq 50 \
    --eval-steps 100 \
    --anneal-lr False \ 
    --cut-off-limit 10 \
    --win-reward 3 \
    --fail-penalty 0.1