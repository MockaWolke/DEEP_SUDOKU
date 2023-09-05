cd /home/felix/Desktop/DEEP_SUDOKU/reinforcement_learning_experiment

python single_action_ppo_script.py \
    --total-timesteps 5000000 \
    --gym-id Sudoku-nostop0 \
    --exp-name Nonestop_ENV_Shared_right_rewards \
    --mask-actions True \
    --agent SpecialSoftmax \
    --eval-freq 50 \
    --eval-steps 100 \
    --anneal-lr True \
    --upper-bound-missing-digits 5 \
    --win-reward 3 \
    --fail-penalty 1 \