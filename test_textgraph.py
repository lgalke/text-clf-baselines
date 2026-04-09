from collections import Counter
from itertools import chain
from textgraph import TextGraph, count_ww_dw, count_dw

def prepare_dummy_data():
    corpus = [
            "The quick brown fox jumps over the lazy dog",
            "The fox is quick",
            "The dog is lazy",
            "The fox is brown",
            "The color of the dog is unknown"
            ] 
    corpus = [s.lower().split() for s in corpus]
    cnt = Counter(chain.from_iterable(corpus))
    words, __counts = zip(*cnt.most_common())
    vocab = {word: idx for idx, word in enumerate(words)}
    data = [[vocab[t] for t in doc] for doc in corpus]
    vocab_size = len(vocab)
    return data, vocab

def test_textgraph_basic():
    data, vocab = prepare_dummy_data()
    tg = TextGraph(len(vocab), window_size=3, format='csr')
    adj = tg.fit_transform(data)

    # They co-occur frequently, should have pmi > 0
    assert adj[vocab['fox'], vocab['quick']] > 0
    assert adj[vocab['dog'], vocab['lazy']] > 0
    assert adj[vocab['the'], vocab['is']] > 0
    # They never co-occur
    assert adj[vocab['unknown'], vocab['quick']] < 1e-8

def test_textgraph_padding():
    data, vocab = prepare_dummy_data()
    pad = len(vocab)
    vocab_size = len(vocab) + 1
    maxlen = max(len(seq) for seq in data)
    padded_data = [seq + (maxlen - len(seq)) * [pad] for seq in data]
    tg = TextGraph(vocab_size, window_size=3, padding_idx=pad, format='csr')

    adj_nopad = tg.fit_transform(data)
    adj_pad = tg.fit_transform(padded_data)

    assert (adj_pad[pad, :] - adj_nopad[pad, :]).sum() < 1e-8, "Padding treated wrongly"
    assert (adj_pad[:, pad] - adj_nopad[:, pad]).sum() < 1e-8, "Padding treated wrongly"

def test_textgraph_fit_transform():
    data, vocab = prepare_dummy_data()
    tg1 = TextGraph(len(vocab), window_size=3, format='csr')
    adj1 = tg1.fit_transform(data)

    tg2 = TextGraph(len(vocab), window_size=3, format='csr')
    tg2.fit(data)
    adj2 = tg2.transform(data)


    assert (adj1.toarray() - adj2.toarray() < 1e-8).all()

def test_count_ww_dw_diag_sum():
    data, vocab = prepare_dummy_data()
    ww, dw = count_ww_dw(data, len(vocab), window_size=3)

    n_raw = len(list(chain.from_iterable(data)))
    n_diag = ww.diagonal().sum()
    assert n_diag == n_raw

def test_count_ww_dw_parallel():
    data, vocab = prepare_dummy_data()

    # Ground truth
    ww_true, dw_true = count_ww_dw(data, len(vocab), 3)

    # Parallel impl
    for n_jobs in [1,2,3,4,8]:
        tg = TextGraph(len(vocab), window_size=3, format='csr', n_jobs=n_jobs)
        ww, dw = tg._count_ww_dw_parallel(data)

        assert (ww.toarray() - ww_true.toarray() < 1e-8).all()
        assert (dw.toarray() - dw_true.toarray() < 1e-8).all()

def test_count_dw_parallel():
    data, vocab = prepare_dummy_data()

    # Ground truth
    dw_true = count_dw(data, len(vocab))

    # Parallel impl
    for n_jobs in [1,2,3,4,8]:
        tg = TextGraph(len(vocab), window_size=3, format='csr', n_jobs=n_jobs)
        dw = tg._count_dw_parallel(data)
        assert (dw.toarray() - dw_true.toarray() < 1e-8).all()
        print("1 has worked")


def test_parallel_fit_transform():
    data, vocab = prepare_dummy_data()
    tg1 = TextGraph(len(vocab), window_size=3, format='csr', n_jobs=1)
    adj1 = tg1.fit_transform(data)

    tg2 = TextGraph(len(vocab), window_size=3, format='csr', n_jobs=2)
    adj2 = tg2.fit_transform(data)
    assert (adj2.toarray() - adj1.toarray() < 1e-8).all()

    tg5 = TextGraph(len(vocab), window_size=3, format='csr', n_jobs=5)
    adj5 = tg5.fit_transform(data)
    assert (adj5.toarray() - adj1.toarray() < 1e-8).all()

    tg6 = TextGraph(len(vocab), window_size=3, format='csr', n_jobs=6)
    adj6 = tg6.fit_transform(data)
    assert (adj6.toarray() - adj1.toarray() < 1e-8).all()





