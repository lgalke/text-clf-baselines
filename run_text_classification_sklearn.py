import numpy as np

from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from data import load_data
from transformers import AutoTokenizer

ngram_range = (1,1)
tokenizer_name = 'bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

# VALID_DATASETS = [ '20ng', 'R8', 'R52', 'ohsumed', 'mr'] + ['TREC', 'wiki']
results = {}
for dataset_name in [ '20ng', 'R8', 'R52', 'ohsumed', 'mr']:

    raw_docs, raw_labels, train_mask, test_mask = load_data(dataset_name, tokenizer, n_jobs=4, raw=True)

    raw_docs = np.array(raw_docs, dtype=object)
    train_mask = train_mask.numpy()
    test_mask = test_mask.numpy()

    le = LabelEncoder()
    labels = le.fit_transform(raw_labels)

    y_train = labels[train_mask]
    y_test = labels[test_mask]

    tfidf = TfidfVectorizer(analyzer="word", tokenizer=tokenizer.tokenize,
                            ngram_range=ngram_range)
    #tfidf = TfidfVectorizer(analyzer=tokenizer.tokenize, ngram_range=(1,1))

    x_train = tfidf.fit_transform(raw_docs[train_mask])  # Only fit idf on train
    x_test = tfidf.transform(raw_docs[test_mask])

    svm = LinearSVC()

    svm.fit(x_train, y_train)

    y_pred = svm.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(dataset_name, accuracy)
    results[dataset_name] = accuracy

print(f"ngram_range = {ngram_range} ")
print(results)
