import torch
import torch.nn as nn
from typing import List

"""Embedding Initialization from Names


    bert_classifier.bert.embeddings.word_embeddings

"""


def init_classifier_from_embedded_class_names(
    model: nn.Module, tokenizer, class_names: List[str]
):
    """*ForSequenceClassification"""
    # word_embeddings = model.bert.embeddings.word_embeddings
    # more general:
    assert hasattr(model, "get_input_embeddings")
    assert hasattr(model, "classifier")
    print("EIN Init called with class names:")
    print(class_names[:20])

    word_embeddings = model.get_input_embeddings()
    classifier = model.classifier

    emb_by_class = []

    for class_name in class_names:
        token_ids = tokenizer.encode(class_name, return_tensors="pt")  # [1, seqlen]
        emb = word_embeddings(token_ids)  # [1, seqlen, emb_size]
        emb_mean = emb.mean(dim=1)  # [1, emb_size]
        emb_by_class.append(emb_mean)

    emb_by_class = torch.cat(emb_by_class, dim=0)

    # Now set output embeddings to thingy
    print("Final Emb size", emb_by_class.size())
    print("Classifier weight size", classifier.weight.data.size())

    classifier.weight.data = emb_by_class
    print("Inited.")
