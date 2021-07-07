#!/usr/bin/env bash
MODEL_TYPE="mlp"
# MODEL_NAME_OR_PATH="distilbert-base-uncased"
TOKENIZER_NAME="bert-base-uncased"
BATCH_SIZE=16
MLP_NUM_LAYERS=2
EPOCHS=100
RESULTS_FILE="results_mlp-2.csv"

set -e

for seed in 1 2 3 4 5; do
	for DATASET in "20ng" "R8" "R52" "ohsumed" "mr"; do
		python3 run_text_classification.py --model_type "$MODEL_TYPE" --tokenizer_name "$TOKENIZER_NAME" \
			--batch_size $BATCH_SIZE --learning_rate "0.001" --mlp_num_layers "$MLP_NUM_LAYERS" \
			--epochs $EPOCHS --num_workers 4 --results_file "$RESULTS_FILE" "$DATASET"
		done
	done

