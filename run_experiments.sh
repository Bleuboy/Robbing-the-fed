#!/bin/zsh

MODEL_TYPES=("resnet18" "resnet18" "resnet18")
BATCH_SIZES=(1 4 8)
EPOCHS=(1 2 3)
MAX_GRAD_NORMS=(1.0 1.1 1.3)
NOISE_MULTIPLIERS=(1.1 1.2 2)

for model in "${MODEL_TYPES[@]}"; do
  for batch in "${BATCH_SIZES[@]}"; do
    for epoch in "${EPOCHS[@]}"; do
      for max_grad_norm in "${MAX_GRAD_NORMS[@]}"; do
        for noise_multiplier in "${NOISE_MULTIPLIERS[@]}"; do
          python3 training_opacus.py --batch_size "$batch" --EPOCHS "$epoch" --MAX_GRAD_NORM "$max_grad_norm" --NOISE_MULTIPLIER "$noise_multiplier" --model "$model"
        done
      done
    done
  done
done
