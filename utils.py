import torch

def calc_accuracy(logits, true_labels):
    __max_vals, max_indices = torch.max(logits, 1)
    acc = (max_indices == true_labels).sum().float() / true_labels.size(0)
    return acc
