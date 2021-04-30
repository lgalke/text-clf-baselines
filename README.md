# gnlp

**Key point:** Re-run experiments on TextGCN datasets, but compare TextGCN with
some transformers and plain MLPs.

## Get up and running

1. Make sure that the data is placed into a subfolder `./data` in the exact same
   format as here: [https://github.com/iworldtong/text_gcn.pytorch/tree/master/data](https://github.com/iworldtong/text_gcn.pytorch/tree/master/data)

2. Double check for other static paths, I already adjusted `CACHE_DIR` in
   `data.py` to be relative: `tmp/cache`, but their might be others.

3. Check for dependencies (unfortunately, there is no requirements.txt file),
   but you need `numpy`, `torch`, `torch_geometric`, `transformers`, `tqdm`, `joblib`


## Models

In `models.py`, there are some models implemented, that are not coming directly from huggingface's transformers library.

- MLP 
- GCN (for TextGCN), may run into mem issues when on GPU
- TransformerForNodeClassification (early attempt for some hybrid model, yet not finished, you can ignore this)

## Data

In `data.py`, you find the `load_data()` function which, does the data loading
(format of TextGCN paper) and construction of graph, and then, outputs
`torch_geometric.data.Data` while caching the computation.

Valid datasets are: `[ '20ng', 'R8', 'R52', 'ohsumed', 'mr'] + ['TREC', 'wiki']`
from TextGCN even though TREC and wiki are not reported in their paper.

## Text graph

Same files as I've send you earlier: `textgraph.py` and `test_textgraph.py`.

## Running experiments

The script `run_text_classification.py` is the main entry point for running an experiment.
It distinguishes two paths `run_axy_model` and `run_xy_model`, depending on whether you need adjacency information or not.
Train test split is used as-is from the datasets.

## Some examples

```bash
# DistilBERT
python3 run_text_classification.py --model_type distilbert --model_name_or_path distilbert-base-uncased --results_file quick-test-2020-03.txt ohsumed
# MLP with DistilBERT Tokenizer
python3 run_text_classification.py --model_type mlp --results_file quick-test-2020-03.txt ohsumed --tokenizer_name distilbert-base-uncased
# TextGCN with DistilBERT Tokenizer
python3 run_text_classification.py --model_type textgcn --results_file quick-test-2020-03.txt ohsumed --tokenizer_name distilbert-base-uncased
```
