#!/usr/bin/env bash
MODEL_TYPE="mlp"
MODEL_NAME_OR_PATH="path/to/glove.42B.300d.txt"
BATCH_SIZE=16
EPOCHS=100
RESULTS_FILE="results_glove42b_mlp.csv"
LEARNING_RATE="0.001"

# Stop on error
set -e
for seed in 1 2 3 4 5; do
	for DATASET in "20ng" "R8" "R52" "ohsumed" "mr"; do
		python3 run_text_classification.py --model_type "$MODEL_TYPE" --model_name_or_path "$MODEL_NAME_OR_PATH" \
			--batch_size $BATCH_SIZE --learning_rate $LEARNING_RATE --mlp_num_layers 2 --mlp_embedding_dropout "0.0" \
			--epochs $EPOCHS --num_workers 4 --results_file "$RESULTS_FILE" "$DATASET"
	done
done

