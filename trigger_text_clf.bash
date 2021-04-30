#!/usr/bin/env bash
MODEL_TYPE="distilbert"
MODEL_NAME_OR_PATH="distilbert-base-uncased"
BATCH_SIZE=20
GRADIENT_ACCUMULATION_STEPS=10
EPOCHS=100
RESULTS_FILE="results_bs20x10.csv"

# for DATASET in "wiki"; do
# for DATASET in "20ng" "R8" "R52" "ohsumed" "mr" "TREC" "wiki"; do
for DATASET in "R8" "wiki"; do
	python3 run_text_classification.py --model_type "$MODEL_TYPE" --model_name_or_path "$MODEL_NAME_OR_PATH" \
		--batch_size $BATCH_SIZE --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
		--epochs $EPOCHS --num_workers 4 --results_file "$RESULTS_FILE" "$DATASET"
	done
