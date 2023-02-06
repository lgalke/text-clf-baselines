import os
import numpy as np

from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from data import load_data
from transformers import AutoTokenizer

import openai

openai.organization = "org-MHqlKDvTANBEASvMZSUJGnvZ"
openai.api_key = os.getenv("OPENAI_API_KEY")

results = {}
for dataset_name in ['R8']:

    raw_docs, raw_labels, train_mask, test_mask = load_data(dataset_name, tokenizer, n_jobs=4, raw=True)

    raw_docs = np.array(raw_docs, dtype=object)
    raw_labels = np.array(raw_labels, dtype=object)
    train_mask = train_mask.numpy()
    test_mask = test_mask.numpy()


    test_docs = raw_docs[test_mask]
    test_labels = raw_labels[test_mask]

    le = LabelEncoder().fit(raw_labels)
    classes = le.classes_
    prompt_prefix = f"Classify the following text into one of the classes {classes}:"

    N_total = 0
    N_true_positive = 0

    for test_doc, true_label in zip(test_docs,test_labels):
        prompt = prompt_prefix + '\n\n' + test_doc
        print(prompt)
        result = openai.Completion.create(
            model="text-davinci-003",
            prompt=prompt,
            temperature=0,
            stop="",
        )
        print("\nGPT-3:", result)

        result = result.strip().lower()

        if result == true_label:
            N_true_positive += 1

        N_total += 1

        input()

        if N_total > 5:
            break

    accuracy = N_true_positive / N_total
