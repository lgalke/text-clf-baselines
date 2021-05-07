import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as tg
import tokenizers


# def collate_encoding_for_mlp(examples):
#     encodings, labels = zip(*examples)
#     batch = tokenizers.Encoding.merge(encodings, growing_offsets=True)
#     # we dont need end positions
#     offsets, __ = zip(*batch.offsets)

#     offset = 0
#     flat_docs, offsets, labels = [], [], []
#     for doc, label in list_of_samples:

#         offsets.append(offset)
#         flat_docs.extend(doc)
#         labels.append(label)
#         offset += len(doc)

#     return torch.tensor(batch.ids), torch.tensor(offsets), torch.tensor(labels)



def collate_for_mlp(list_of_samples):
    """ Collate function that creates batches of flat docs tensor and offsets """
    offset = 0
    flat_docs, offsets, labels = [], [], []
    for doc, label in list_of_samples:
        if isinstance(doc, tokenizers.Encoding):
            doc = doc.ids
        offsets.append(offset)
        flat_docs.extend(doc)
        labels.append(label)
        offset += len(doc)
    return torch.tensor(flat_docs), torch.tensor(offsets), torch.tensor(labels)

class MLP(nn.Module):
    """Simple MLP"""
    def __init__(self, vocab_size, num_classes,
                 hidden_size=1024, hidden_act='relu',
                 dropout=0.5, mode='mean'):
        nn.Module.__init__(self)
        self.embed = nn.EmbeddingBag(vocab_size, hidden_size,
                                     mode=mode)
        self.lin = nn.Linear(hidden_size, num_classes)
        self.act = getattr(F, hidden_act)
        self.drop = dropout if dropout else None
        self.loss_function = nn.CrossEntropyLoss()

    def forward(self, input, offsets, labels=None):
        h = self.embed(input, offsets)
        h = self.act(h)
        if self.drop:
            h = F.dropout(h, p=self.drop, training=self.training)
        logits = self.lin(h)
        if labels is not None:
            loss = self.loss_function(logits, labels)
            return loss, logits
        return logits


class WordEmbeddingMLP(nn.Module):
    """ Word Embedding + MLP """
    def __init__(self, embeddings, num_classes,
                 hidden_size=1024, hidden_act='relu',
                 dropout=0.5, mode='mean'):
        nn.Module.__init__(self)
        self.embed = nn.EmbeddingBag.from_pretrained(embeddings, freeze=True)
        self.lin1 = nn.Linear(embeddings.size(1), hidden_size)
        self.lin2 = nn.Linear(hidden_size, num_classes)
        self.act = getattr(F, hidden_act)
        self.drop = dropout if dropout else None
        self.loss_function = nn.CrossEntropyLoss()

    def forward(self, input, offsets, labels=None):
        # Embedding (frozen)
        embeds = self.embed(input, offsets)
        # MLP
        # input-to-hidden
        h = self.lin1(embeds)
        h = self.act(h)
        if self.drop:
            h = F.dropout(h, p=self.drop, training=self.training)
        # hidden-to-output
        logits = self.lin2(h)
        if labels is not None:
            loss = self.loss_function(logits, labels)
            return loss, logits
        return logits



class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_labels,
                 dropout=None, activation=F.relu):
        super(GCN, self).__init__()
        self.conv1 = tg.nn.GCNConv(in_channels, hidden_channels)
        self.act = activation
        self.conv2 = tg.nn.GCNConv(hidden_channels, num_labels)
        self.drop = nn.Dropout(dropout) if dropout else None
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x, edge_index, edge_weight, labels=None, mask=None):
        h = self.conv1(x, edge_index, edge_weight)
        h = self.act(h)
        if self.drop:
            h = self.drop(h)
        logits = self.conv2(h, edge_index, edge_weight)
        if mask is not None:
            logits = logits[mask]

        outputs = (logits, )  # Always output tuple

        if labels is not None:
            if mask is not None:
                labels = labels[mask]
            loss = self.loss_fn(logits, labels)
            outputs = (loss, ) + outputs

        return outputs



class TransformerForNodeClassification(nn.Module):
    def __init__(self, transformer_model, in_channels, hidden_channels, out_channels,
            activation=F.relu, dropout=None):
        super(TransformerForNodeClassification, self).__init__()
        self.transformer = transformer_model
        self.conv1 = tg.nn.GraphConv(in_channels, hidden_channels)
        self.act = act
        self.conv2 = tg.nn.GraphConv(hidden_channels, out_channels)
        self.drop = nn.Dropout(dropout) if dropout else None
        self.loss_fn = nn.CrossEntropyLoss()


    def forward(self,
            input_ids=None,
            attention_mask=None,
            head_mask=None,
            inputs_embeds=None,
            edge_index=None,
            edge_weight=None,
            labels=None,
            mask=None):
        transformer_outputs = self.transformer(input_ids=input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask, inputs_embeds=inputs_embeds)
        last_hidden_state = transformer_outputs[0]  # Last hidden state is first elem of output tuple
        h = last_hidden_state[:, 0]  # use pooled output (as in *ForSequenceClassification)
        # h = last_hidden_state.mean(1)  # Avg over sequence
        self.conv1(h, edge_index, edge_weight)
        h = self.act(h)
        if self.drop:
            h = self.drop(h)
        logits = self.conv2(h, edge_index, edge_weight) # (bs, num_labels)
        if mask is not None:
            logits = logits[mask]

        outputs = (logits, ) + transformer_outputs

        if labels is not None:
            if mask is not None:
                labels = labels[mask]
            loss = self.loss_fn(logits, labels)
            outputs = (loss, ) + outputs

        return outputs
