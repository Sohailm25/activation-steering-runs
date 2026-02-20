#!/bin/bash
cd "$(dirname "$0")/.."

echo "Launching 14B self-control seed=43"
python3 -m modal run --detach infrastructure/v4_transfer_b1.py --source qwen-14b --target qwen-14b --multiplier 10 --seed 43 &
sleep 5

echo "Launching 32B self-control seed=43"
python3 -m modal run --detach infrastructure/v4_transfer_b1.py --source qwen-32b --target qwen-32b --multiplier 15 --seed 43 &
sleep 5

echo "Launching 14B->32B transfer seed=43"
python3 -m modal run --detach infrastructure/v4_transfer_b1.py --source qwen-14b --target qwen-32b --multiplier 15 --seed 43 &
sleep 5

echo "Launching 32B->14B transfer seed=43"
python3 -m modal run --detach infrastructure/v4_transfer_b1.py --source qwen-32b --target qwen-14b --multiplier 10 --seed 43 &
sleep 5

wait
echo "All 4 seed=43 jobs launched"
