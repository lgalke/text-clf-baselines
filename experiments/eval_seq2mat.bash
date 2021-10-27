#!/usr/bin/env bash
MODEL_TYPE="seq2mat"
MODEL_NAME_OR_PATH="./seq2mat-model/"
# SEQ2MAT_CONFIG="./seq2mat-config/seq2mat_HB_mlp_comparable.json"
SEQ2MAT_CONFIG="./seq2mat-config/seq2mat_hybrid_bidirectional_diffcat.json"
BATCH_SIZE=64
GRADIENT_ACCUMULATION_STEPS=1
EPOCHS=10
RESULTS_FILE="results_seq2mat.csv"
LEARNING_RATE="0.001"
MAX_LENGTH=512

# Stop on error
set -e
# for seed in 1 2 3 4 5; do
for seed in 1; do
	for DATASET in "20ng" "R8" "R52" "ohsumed" "mr"; do
		python3 run_text_classification.py --model_type "$MODEL_TYPE" --model_name_or_path "$MODEL_NAME_OR_PATH" --seq2mat_config "$SEQ2MAT_CONFIG" --seq2mat_max_length $MAX_LENGTH \
			--batch_size $BATCH_SIZE --learning_rate "$LEARNING_RATE" --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
			--epochs $EPOCHS --num_workers 4 --results_file "$RESULTS_FILE" "$DATASET"
	done
done

