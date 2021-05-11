#!/usr/bin/env bash
MODEL_TYPE="mlp"
# MODEL_NAME_OR_PATH="distilbert-base-uncased"
TOKENIZER_NAME="bert-base-uncased"
BATCH_SIZE=16
GRADIENT_ACCUMULATION_STEPS=10
EPOCHS=100
RESULTS_FILE="results_mlp.csv"

for DATASET in "20ng" "R8" "R52" "ohsumed" "mr"; do
	python3 run_text_classification.py --model_type "$MODEL_TYPE" --tokenizer_name "$TOKENIZER_NAME" \
		--batch_size $BATCH_SIZE --learning_rate "0.001"\
		--epochs $EPOCHS --num_workers 4 --results_file "$RESULTS_FILE" "$DATASET"
	done

for DATASET in "20ng" "R8" "R52" "ohsumed" "mr"; do
	python3 run_text_classification.py --model_type "$MODEL_TYPE" --tokenizer_name "$TOKENIZER_NAME" \
		--batch_size $BATCH_SIZE --learning_rate "0.001"\
		--epochs $EPOCHS --num_workers 4 --results_file "$RESULTS_FILE" "$DATASET"
	done

for DATASET in "20ng" "R8" "R52" "ohsumed" "mr"; do
	python3 run_text_classification.py --model_type "$MODEL_TYPE" --tokenizer_name "$TOKENIZER_NAME" \
		--batch_size $BATCH_SIZE --learning_rate "0.001"\
		--epochs $EPOCHS --num_workers 4 --results_file "$RESULTS_FILE" "$DATASET"
	done
