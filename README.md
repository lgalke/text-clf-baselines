# Forget me not: A Gentle Reminder to Mind the Simple Multi-Layer Perceptron Baseline for Text Classification -- Code for Experiments

## Get up and running

1. Download the data folder from the [TextGCN repository](https://github.com/yao8839836/text_gcn) and make sure that the data is placed into a subfolder `./data` in the exact same directory structure. *This is not our repository, so this does **not** de-anonymize us.*

2. Check for static paths such as `CACHE_DIR`in `run_text_classification.py` and the path to GloVe vectors in the /experiments dir (don't worry, they're anonymized)

3. Check for dependencies (unfortunately, there is no requirements.txt file),
   but you need `numpy`, `torch`, `transformers`, `tqdm`, `joblib`, `tokenizers`, and `scikit-learn`. We tried to clean-up torch-geometric dependency as it is not needed for the paper's experiments.


## Code overview

- In `models.py`, you find our implementation for the SimpleMLP.
- In `data.py`, you find the `load_data()` function which, does the data loading. Valid datasets are: `[ '20ng', 'R8', 'R52', 'ohsumed', 'mr']
- In `tokenization.py` you find our tokenizer implementation for the GloVe model. For other models, use BERT's tokenizer and vocab
- The code contains some artefacts from creating a textgraph ourselves, but in the end we did not run own experiments that needed it.

## Running experiments

The script run\_text\_classification.py is the main entry point for running an experiment.
Train test split is used as-is from the datasets of TextGCN.
Within the experiments folder, you find the bash scripts that we used for the experiments.
