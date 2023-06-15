import os
import numpy as np

from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from data import load_data

import openai

openai.organization = "org-MHqlKDvTANBEASvMZSUJGnvZ"
openai.api_key = os.getenv("OPENAI_API_KEY")

results = {}


def make_example(doc, label=None):
    s = f"Text: {doc}\n"
    if label is not None:
        s += f"Class: {label}\n"
    else:
        s += "Class: "
    return s


def select_few_shots(train_docs, train_labels):
    assert len(train_docs) == len(train_labels)
    train_labels_list = list(train_labels)
    few_shots = ""
    for label in np.unique(train_labels):
        idx = train_labels_list.index(label)
        few_shots += make_example(train_docs[idx], train_labels[idx])
    return few_shots


for dataset_name in ["R8"]:
    raw_docs, raw_labels, train_mask, test_mask = load_data(
        dataset_name, None, n_jobs=4, raw=True
    )

    raw_docs = np.array(raw_docs, dtype=object)
    raw_labels = np.array(raw_labels, dtype=object)
    train_mask = train_mask.numpy()
    test_mask = test_mask.numpy()

    test_docs = raw_docs[test_mask]
    test_labels = raw_labels[test_mask]

    le = LabelEncoder().fit(raw_labels)
    classes = le.classes_
    task_desc = f"Classify text into one of the classes {classes}.\n"

    N_total = 0
    N_true_positive = 0

    few_shots = select_few_shots(raw_docs[train_mask], raw_labels[train_mask])

    for test_doc, true_label in zip(test_docs, test_labels):
        prompt = task_desc + few_shots + make_example(test_doc)
        print(prompt)
        result = openai.Completion.create(
            model="text-davinci-003",
            prompt=prompt,
            temperature=0,
            stop="",
        )
        print("\nGPT-3:", result)

        pred_label = result["choices"][0]["text"].strip()

        if pred_label.lower() == true_label.lower():
            N_true_positive += 1
            print("\nEvaluation", "Correct!")
        else:
            print(f"\nEvaluation: Wrong! True label was {true_label}")

        N_total += 1

        input()

        if N_total > 10:
            break

    accuracy = N_true_positive / N_total
    print(f"Accuracy: {accuracy:2.f} (N={N_total})")
