# Bag-of-Words vs. Graph vs. Sequence in Text Classification: Questioning the Necessity of Text-Graphs and the Surprising Strength of a Wide MLP

*Code for the experiments*

[Link to ACL 2022 paper](https://github.com/lgalke/text-clf-baselines)

If you use this code for your research, please consider citing:

```bibtex
@inproceedings{galke-scherp-2022-bag,
    title = "Bag-of-Words vs. Graph vs. Sequence in Text Classification: Questioning the Necessity of Text-Graphs and the Surprising Strength of a Wide {MLP}",
    author = "Galke, Lukas  and
      Scherp, Ansgar",
    booktitle = "Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = may,
    year = "2022",
    address = "Dublin, Ireland",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.acl-long.279",
    pages = "4038--4051",
    abstract = "Graph neural networks have triggered a resurgence of graph-based text classification methods, defining today{'}s state of the art. We show that a wide multi-layer perceptron (MLP) using a Bag-of-Words (BoW) outperforms the recent graph-based models TextGCN and HeteGCN in an inductive text classification setting and is comparable with HyperGAT. Moreover, we fine-tune a sequence-based BERT and a lightweight DistilBERT model, which both outperform all state-of-the-art models. These results question the importance of synthetic graphs used in modern text classifiers. In terms of efficiency, DistilBERT is still twice as large as our BoW-based wide MLP, while graph-based models like TextGCN require setting up an $\mathcal{O}(N^2)$ graph, where $N$ is the vocabulary plus corpus size. Finally, since Transformers need to compute $\mathcal{O}(L^2)$ attention weights with sequence length $L$, the MLP models show higher training and inference speeds on datasets with long sequences.",
}
```


## Get up and running

1. Download the data folder from the [TextGCN repository](https://github.com/lgalke/text_gcn) (forked for archival purposes) and make sure that the data is placed into a subfolder `./data` in the exact same directory structure.

2. Check for static paths such as `CACHE_DIR`in `run_text_classification.py` and the path to GloVe vectors in the /experiments dir

3. Install dependencies via `pip install -r requirements.txt`, preferably in a virtual environment.

## Code overview

- In `models.py`, you find our implementation of the WideMLP.
- In `data.py`, you find the `load_data()` function which, does the data loading. Valid datasets are: `[ '20ng', 'R8', 'R52', 'ohsumed', 'mr']
- In `tokenization.py` you find our tokenizer implementation for the GloVe model. For other models, use BERT's tokenizer and vocab
- The code contains some artefacts from creating a textgraph ourselves, but in the end we did not run own experiments that needed it.

## Running experiments

The script run\_text\_classification.py is the main entry point for running an experiment.
Train test split is used as-is from the datasets of TextGCN.
Within the experiments folder, you find the bash scripts that we used for the experiments from the paper.
