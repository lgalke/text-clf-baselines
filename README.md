# Bag-of-Words vs. Graph vs. Sequence in Text Classification: Questioning the Necessity of Text-Graphs and the Surprising Strength of a Wide MLP

*Code for the experiments*

If you use this code for your research, please consider citing:

```bibtex
@misc{galke2021forget,
      title={Forget me not: A Gentle Reminder to Mind the Simple Multi-Layer Perceptron Baseline for Text Classification}, 
      author={Lukas Galke and Ansgar Scherp},
      year={2021},
      eprint={2109.03777},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

## Get up and running

1. Download the data folder from the [TextGCN repository](https://github.com/lgalke/text_gcn) (forked for archival purposes) and make sure that the data is placed into a subfolder `./data` in the exact same directory structure.

2. Check for static paths such as `CACHE_DIR`in `run_text_classification.py` and the path to GloVe vectors in the /experiments dir

3. Install dependencies via `pip install -r requirements.txt`, preferably in a virtual environment.

## Code overview

- In `models.py`, you find our implementation for the SimpleMLP.
- In `data.py`, you find the `load_data()` function which, does the data loading. Valid datasets are: `[ '20ng', 'R8', 'R52', 'ohsumed', 'mr']
- In `tokenization.py` you find our tokenizer implementation for the GloVe model. For other models, use BERT's tokenizer and vocab
- The code contains some artefacts from creating a textgraph ourselves, but in the end we did not run own experiments that needed it.

## Running experiments

The script run\_text\_classification.py is the main entry point for running an experiment.
Train test split is used as-is from the datasets of TextGCN.
Within the experiments folder, you find the bash scripts that we used for the experiments.
