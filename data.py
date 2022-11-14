import os.path as osp
import random

import numpy as np
import torch
import zipfile
from joblib import Memory
from tqdm import tqdm

from textgraph import TextGraph

CACHE_DIR = 'tmp/cache'
MEMORY = Memory(CACHE_DIR, verbose=2)
VALID_DATASETS = [ '20ng', 'R8', 'R52', 'ohsumed', 'mr'] + ['TREC', 'wiki']



@MEMORY.cache
def load_word_vectors(path, unk_token=None):
    vocab = dict()
    vectors = []
    with open(path, mode='r') as myfile:
        for i, line in tqdm(enumerate(myfile)):
            word, *vector_str = line.strip().split(' ')
            if len(vector_str) == 1:
                print(f"[load_word_vectors] Ignoring row {i+1}: {line}")
                continue

            # Parse word vector
            vector = torch.tensor([float(val) for val in vector_str])

            vocab[word] = len(vocab)
            vectors.append(vector)


    if unk_token:
        print(f"Adding UNK token: '{unk_token}'")
        assert isinstance(unk_token, str), "Unk token needs to be str"
        assert unk_token not in vocab, "Unk token may not be in vocab"
        vocab[unk_token] = len(vocab)
        vectors.append(torch.zeros_like(vectors[0]))


    embedding = torch.stack(vectors)

    return vocab, embedding


@MEMORY.cache(ignore=['n_jobs'])
def load_data(key, tokenizer, max_length=None, construct_textgraph=False, n_jobs=1,
              force_lowercase=False, raw=False):
    assert key in VALID_DATASETS, f"{key} not in {VALID_DATASETS}"
    print("Loading raw documents")
    with open(osp.join('data', 'corpus', key+'.txt'), 'rb') as f:
        raw_documents = [line.strip().decode('latin1') for line in tqdm(f)]

    N = len(raw_documents)

    # print("First few raw_documents", *raw_documents[:5], sep='\n')

    labels = []
    train_mask, test_mask = torch.zeros(N, dtype=torch.bool), torch.zeros(N, dtype=torch.bool)
    print("Loading document metadata...")
    doc_meta_path = osp.join('data', key+'.txt')
    with open(doc_meta_path, 'r') as f:
        for idx, line in tqdm(enumerate(f)):
            __name, train_or_test, label = line.strip().split('\t')
            if 'test' in train_or_test:
                test_mask[idx] = True
            elif 'train' in train_or_test:
                train_mask[idx] = True
            else:
                raise ValueError("Doc is neither train nor test:"
                                 + doc_meta_path + ' in line: ' + str(idx+1))
            labels.append(label)

    assert len(labels) == N
    # raw_documents, labels, train_mask, test_mask defined

    if raw:
        return raw_documents, labels, train_mask, test_mask

    if max_length:
        print(f"Encoding documents with max_length={max_length}...")
        # docs = [tokenizer.encode(raw_doc, max_length=max_length) for raw_doc in raw_documents]
        # docs = tokenizer(raw_documents, truncation=True, max_length=max_length)

        # Now use truncation=True (continued experiments with seq2mat)
        docs = [tokenizer.encode(raw_doc, truncation=True, max_length=max_length) for raw_doc in raw_documents]
    else:
        print(f"Encoding documents without max_length")
        docs = [tokenizer.encode(raw_doc) for raw_doc in raw_documents]

    print("Encoding labels...")
    label2index = {label: idx for idx, label in enumerate(set(labels))}
    label_ids = [label2index[label] for label in tqdm(labels)]


    if not construct_textgraph:
        return docs, label_ids, train_mask, test_mask, label2index

    # ONLY NOW we need geometric
    import torch_geometric
    vocab_size, pad_token_id = tokenizer.vocab_size, tokenizer.pad_token_id
    textgraph = TextGraph(vocab_size, window_size=20,
                   padding_idx=pad_token_id, format='coo',
                   n_jobs=n_jobs, verbose=10)
    train_docs = np.asarray(docs)[train_mask]
    print("Fitting textgraph...")
    textgraph.fit(train_docs)
    print("Transforming docs...")
    adj = textgraph.transform(docs)
    edge_index = torch.tensor([adj.row, adj.col], dtype=torch.long)
    edge_attr = torch.tensor(adj.data).unsqueeze(1)

    if max_length is None:
        max_length = max(len(doc) for doc in docs)

    print("Preparing feats...")
    # Right padding
    x_docs = torch.tensor([doc + (max_length - len(doc)) * [pad_token_id] for doc in docs])
    x_words = torch.LongTensor(vocab_size, max_length).fill_(pad_token_id)
    x_words[:, 0] = torch.arange(vocab_size)
    x = torch.cat([x_words, x_docs], dim=0)

    print("Preparing labels")
    y_docs = torch.tensor(label_ids)
    dummy_label_id = -1
    y_words = torch.LongTensor(vocab_size).fill_(dummy_label_id)
    y = torch.cat([y_words, y_docs], dim=0)

    print("Preparing masks")
    train_mask = torch.cat([torch.zeros(vocab_size, dtype=torch.bool),
                            train_mask], dim=0)
    test_mask = torch.cat([torch.zeros(vocab_size, dtype=torch.bool),
                           test_mask], dim=0)
    word_mask = torch.cat([torch.ones(vocab_size, dtype=torch.bool),
                           torch.zeros(x_docs.shape[0], dtype=torch.bool)],
                          dim=0)
    num_nodes = len(docs) + vocab_size
    assert word_mask.size(0) == train_mask.size(0) \
        == test_mask.size(0) == num_nodes
    assert x.size(0) == y.size(0) == num_nodes

    data = torch_geometric.data.Data(x=x,
                                     edge_index=edge_index,
                                     edge_attr=edge_attr,
                                     y=y,
                                     train_mask=train_mask,
                                     test_mask=test_mask,
                                     word_mask=word_mask,
                                     label2index=label2index,
                                     dummy_label_id=dummy_label_id)

    return data

def shuffle_augment(docs: list, labels: list,
                    factor:float=1.0, random_seed=None):
    assert factor > 0.0
    if random_seed is not None:
        random.seed(random_seed)
    num_augment = int(len(docs) * factor)
    print(f"Generating {num_augment} augmented documents...")

    new_docs = []
    new_labels = []

    for __ in tqdm(range(num_augment)):
        # Draw a document index
        idx = random.sample(range(len(docs)), k=1)[0]
        doc = docs[idx]

        # Shuffle the words of the document (copy)
        perm_doc = random.sample(doc, k=len(doc))

        new_docs.append(perm_doc)  # Save the new document
        new_labels.append(labels[idx])  # Copy over label from origin

    return new_docs, new_labels
