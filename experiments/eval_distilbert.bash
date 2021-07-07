#!/usr/bin/env bash
MODEL_TYPE="distilbert"
MODEL_NAME_OR_PATH="distilbert-base-uncased"
BATCH_SIZE=64
GRADIENT_ACCUMULATION_STEPS=2
EPOCHS=10
RESULTS_FILE="results_distilbert.csv"
LEARNING_RATE="0.00005"

# Stop on error
set -e
for seed in 1 2 3 4 5; do
	for DATASET in "20ng" "R8" "R52" "ohsumed" "mr"; do
		python3 run_text_classification.py --model_type "$MODEL_TYPE" --model_name_or_path "$MODEL_NAME_OR_PATH" \
			--batch_size $BATCH_SIZE --learning_rate "$LEARNING_RATE" --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
			--epochs $EPOCHS --num_workers 4 --results_file "$RESULTS_FILE" "$DATASET"
	done
done

