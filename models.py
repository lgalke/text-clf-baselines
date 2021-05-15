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
                 num_hidden_layers=1,
                 hidden_size=1024, hidden_act='relu',
                 dropout=0.5, idf=None, mode='mean',
                 pretrained_embedding=None, freeze=True):
        nn.Module.__init__(self)
        # Treat TF-IDF mode appropriately
        mode = 'sum' if idf is not None else mode
        self.idf = idf

        # Input-to-hidden (efficient via embedding bag)
        if pretrained_embedding is not None:
            # vocabsize is defined by embedding in this case
            self.embed = nn.EmbeddingBag.from_pretrained(pretrained_embedding, freeze=freeze, mode=mode)
            embedding_size = pretrained_embedding.size(1)
        else:
            assert vocab_size is not None
            self.embed = nn.EmbeddingBag(vocab_size, hidden_size, mode=mode) 
            embedding_size = hidden_size

        self.activation = getattr(F, hidden_act)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList()

        # Hidden-to-hidden
        for i in range(num_hidden_layers - 1):
            if i == 0:
                self.layers.append(nn.Linear(embedding_size, hidden_size))
            else:
                self.layers.append(nn.Linear(hidden_size, hidden_size))

        # Hidden-to-output
        self.layers.append(nn.Linear(hidden_size if self.layers else embedding_size, num_classes))

        # Loss function
        self.loss_function = nn.CrossEntropyLoss()

    def forward(self, input, offsets, labels=None):
        # Use idf weights if present
        idf_weights = self.idf[input] if self.idf is not None else None

        h = self.embed(input, offsets, per_sample_weights=idf_weights)

        if self.idf is not None:
            # In the TF-IDF case: renormalize according to l2 norm
            h = h / torch.linalg.norm(h, dim=1, keepdim=True)

        for layer in self.layers:
            # at least one
            h = self.activation(h)
            h = self.dropout(h)
            h = layer(h)

        if labels is not None:
            loss = self.loss_function(h, labels)
            return loss, h
        return h


class WordEmbeddingMLP(nn.Module):
    """ Word Embedding + MLP 
    DEPRECATED:
        Use MLP() with pretrained_embeddings instead"""
    def __init__(self, embeddings, num_classes,
                 hidden_size=1024, hidden_act='relu',
                 dropout=0.5, mode='mean'):
        nn.Module.__init__(self)
        self.embed = nn.EmbeddingBag.from_pretrained(embeddings, freeze=True, mode=mode)
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
