#!/bin/bash
# Launch all 8 gain calibration runs (4 for 14B, 4 for 32B)
cd "$(dirname "$0")/.."

for model in qwen-14b qwen-32b; do
  for mult in 10 12.5 15 17.5; do
    echo "Launching $model m=$mult"
    python3 -m modal run --detach infrastructure/v4_transfer_b1.py \
      --source $model --target $model --multiplier $mult &
    sleep 5  # stagger to avoid rate limits
  done
done

wait
echo "All 8 calibration jobs launched"
