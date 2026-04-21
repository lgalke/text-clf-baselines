from sklearn.feature_extraction.text import TfidfTransformer
import scipy.sparse as sp
from joblib import Parallel, delayed
"""
We did not use this code for the experiments because we only run MLP and DistilBERT ourselves, which don't need a text-graph.
"""

def count_ww_dw(docs, vocab_size, window_size, padding_idx=None):
    """ Count word cooccurrences """
    ww = sp.dok_matrix((vocab_size, vocab_size))
    dw = sp.dok_matrix((len(docs), vocab_size))
    for doc_ix, doc in enumerate(docs):
        for pos, i in enumerate(doc):
            # Number of sliding windows contatin word i
            dw[doc_ix, i] += 1
            ww[i, i] += 1
            for j in doc[pos+1:pos+window_size+1]:
                # Number of sliding windows that contain
                # both word i and j
                if i != j:  # diagonal saved for raw counts
                    ww[i, j] += 1
                    ww[j, i] += 1 # symmetric
    if padding_idx is not None:
        ww[padding_idx, :] = 0
        ww[:, padding_idx] = 0
        dw[:, padding_idx] = 0
    return ww, dw

def count_dw(docs, vocab_size, padding_idx=None):
    dw = sp.dok_matrix((len(docs), vocab_size))
    for i, doc in enumerate(docs):
        for j in doc:
            dw[i, j] += 1

    if padding_idx is not None:
        dw[:, padding_idx] = 0
    return dw

def word_adj_matrix_from_counts(ww_counts):
    diag = ww_counts.diagonal()
    # Total number of sliding windows
    n = diag.sum()
    print("diag sum", n)

    rec_diag = 1.0 / (1 + diag) # +1 to mitigate zero division

    pmi = ww_counts / n  # Normalize probas
    pmi = pmi.multiply(rec_diag.reshape(1, -1))  # Div cols by diag vals
    pmi = pmi.multiply(rec_diag.reshape(-1, 1))  # Div rows by diag vals
    pmi = pmi.log1p()  # Natural logarithm plus 1 mitigate zeros

    adj = pmi.todok()  # Use dok format to set items

    adj.setdiag(1)  # Fix diagonal to ones
    # Only retain connections between words where pmi is positive
    adj[adj < 0] = 0

    return adj

class TextGraph():
    def __init__(self, vocab_size, window_size=20, padding_idx=None, format='coo',
                 n_jobs=1, verbose=0):
        self.window_size = window_size
        self.vocab_size = vocab_size
        self.format = format
        self.padding_idx = padding_idx

        self.word_adj_matrix = None
        self.tfidf = TfidfTransformer()
        self.n_jobs = n_jobs
        self.verbose = verbose


    def _count_dw_parallel(self, docs):
        job_size = max(1, int(len(docs) / self.n_jobs))

        jobs = []
        for i in range(0, len(docs), job_size):
            jobs.append(docs[i:i+job_size])

        dws = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(delayed(count_dw)(job,
            self.vocab_size, padding_idx=self.padding_idx)
            for job in jobs)

        return sp.vstack(dws)

    def _count_ww_dw_parallel(self, docs):
        job_size = max(1, int(len(docs) / self.n_jobs))

        jobs = []
        for i in range(0, len(docs), job_size):
            jobs.append(docs[i:i+job_size])

        results = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(delayed(count_ww_dw)(job,
            self.vocab_size, self.window_size, padding_idx=self.padding_idx)
            for job in jobs)

        wws, dws = zip(*results)
        return sum(wws), sp.vstack(dws)

    def fit(self, docs):
        """

        Arguments
        ---------

        - `docs` : List[List[int]]
          Tokenized corpus of documents on which pmi matrix and idf is computed
        """
        # Compute pmi matrix
        ww, dw = self._count_ww_dw_parallel(docs)
        self.word_adj_matrix = word_adj_matrix_from_counts(ww)
        self.tfidf.fit(dw)
        return self
 
    def transform(self, docs):
        """

        Arguments
        ---------

        - `docs` : List[List[int]]
          Tokenized corpus of documents to transform
        """
        # count words
        dw = self._count_dw_parallel(docs)

        x = self.tfidf.transform(dw)
        # Combine term-doc with term-term pmi matrix
        # or previously given base adj matrix
        adj = sp.bmat([[self.word_adj_matrix, x.transpose()],
                       [x, None]], format=self.format)
        adj.setdiag(1)
        return adj

    def fit_transform(self, docs):
        ww, dw = self._count_ww_dw_parallel(docs)
        print("Computing pmi matrix")
        self.word_adj_matrix = word_adj_matrix_from_counts(ww)
        print("Fitting tfidf")
        x = self.tfidf.fit_transform(dw)
        # Combine term-doc with term-term pmi matrix
        adj = sp.bmat([[self.word_adj_matrix, x.transpose()],
                       [x, None]], format=self.format)
        adj.setdiag(1)
        return adj 
