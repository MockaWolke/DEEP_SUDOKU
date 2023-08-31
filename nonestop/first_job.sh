python ppo_script.py \
    --exp-name FirstTry \
    --mask-actions True \
    --agent SeperateOnlyConv \
    --eval-freq 50 \
    --eval-steps 100 \
    --anneal-lr False
    --use-random-starting-point False \
    --cut-off-limit 20