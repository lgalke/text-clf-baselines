#!/usr/bin/env bash
MODEL_TYPE="bert"
MODEL_NAME_OR_PATH="bert-base-uncased"
# Effective batch size should be 128
BATCH_SIZE=32
GRADIENT_ACCUMULATION_STEPS=4
EPOCHS=10
RESULTS_FILE="results_bert_10epochs.csv"
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

