"""
File: run_text_classification.py
Author: ANONYMIZED
Email: ANONYMIZED
Github: ANONYMIZED
Description: Run text classification experiments on TextGCN's datasets
"""

import csv
import itertools as it
import logging

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np

import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as tg
import transformers
from joblib import Memory
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange
from transformers import (WEIGHTS_NAME, AdamW, AutoTokenizer, BertConfig,
                          BertForSequenceClassification, BertModel,
                          BertTokenizer, DistilBertConfig,
                          DistilBertForSequenceClassification, DistilBertModel,
                          DistilBertTokenizer, get_linear_schedule_with_warmup)

from transformers import DISTILBERT_PRETRAINED_MODEL_ARCHIVE_LIST,\
    BERT_PRETRAINED_MODEL_ARCHIVE_LIST


from sklearn.metrics import f1_score
from sklearn.feature_extraction.text import TfidfTransformer




from tokenization import build_tokenizer_for_word_embeddings
from data import load_data, load_word_vectors
from models import MLP, collate_for_mlp

try:
    import wandb
    WANDB = True
except ImportError:
    print("WandB not installed, to track experiments: pip install wandb")
    WANDB = False




logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

logger = logging.getLogger(__name__)
USE_CUDA = torch.cuda.is_available()
CACHE_DIR = 'cache/textclf'
MEMORY = Memory(CACHE_DIR, verbose=2)

VALID_DATASETS = [ '20ng', 'R8', 'R52', 'ohsumed', 'mr'] + ['TREC', 'wiki']

# ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in (BertConfig, XLNetConfig, XLMConfig,
#                                                                                 RobertaConfig, DistilBertConfig)), ())

ALL_MODELS = BERT_PRETRAINED_MODEL_ARCHIVE_LIST + DISTILBERT_PRETRAINED_MODEL_ARCHIVE_LIST
MODEL_CLASSES = {
    'bert': (BertConfig, BertForSequenceClassification, BertModel),
    # 'xlnet': (XLNetConfig, XLNetForSequenceClassification, XLNetModel),
    # 'xlm': (XLMConfig, XLMForSequenceClassification, XLMModel),
    # 'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaModel),
    'distilbert': (DistilBertConfig, DistilBertForSequenceClassification, DistilBertModel)
}


def inverse_document_frequency(encoded_docs, vocab_size):
    """ Returns IDF scores in shape [vocab_size] """
    num_docs = len(encoded_docs)
    counts = sp.dok_matrix((num_docs, vocab_size))
    for i, doc in tqdm(enumerate(encoded_docs), desc="Computing IDF"):
        for j in doc:
            counts[i,j] += 1

    tfidf = TfidfTransformer(use_idf=True, smooth_idf=True)

    tfidf.fit(counts)

    return torch.FloatTensor(tfidf.idf_)


def pad(seqs, with_token=0, to_length=None):
    if to_length is None:
        to_length = max(len(seq) for seq in seqs)
    return [seq + (to_length - len(seq)) * [with_token] for seq in seqs]


def get_collate_for_transformer(pad_token_id):
    """ Closure to include padding in collate function """
    def _collate_for_transformer(examples):
        docs, labels = list(zip(*examples))
        input_ids = torch.tensor(pad(docs, with_token=pad_token_id))
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        attention_mask[input_ids == pad_token_id] = 0
        labels = torch.tensor(labels)
        token_type_ids = torch.zeros_like(input_ids)
        return input_ids, attention_mask, token_type_ids, labels
    return _collate_for_transformer


def train(args,  train_data, model, tokenizer):
    if args.model_type == 'mlp':
        # if args.model_name_or_path is not None:
        #     # Embedding case, use optimized merging from tokenizer lib
        #     collate_fn = collate_encoding_for_mlp
        # else:
        #     # Manual collate with offsets
        collate_fn = collate_for_mlp
    else:
        collate_fn = get_collate_for_transformer(tokenizer.pad_token_id)
    train_loader = torch.utils.data.DataLoader(train_data,
                                               collate_fn=collate_fn,
                                               shuffle=True,
                                               batch_size=args.batch_size,
                                               num_workers=args.num_workers,
                                               pin_memory=('cuda' in str(args.device)))

    # len(train_loader) no of batches
    t_total = len(train_loader) // args.gradient_accumulation_steps * args.epochs

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon)
    # scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)
    writer = SummaryWriter()

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_data))
    logger.info("  Num Epochs = %d", args.epochs)
    logger.info("  Batch size  = %d", args.batch_size)
    logger.info("  Total train batch size (w. accumulation) = %d",
                   args.batch_size * args.gradient_accumulation_steps)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(args.epochs, desc="Epoch")
    for epoch in train_iterator:
        epoch_iterator = tqdm(train_loader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            if args.model_type == 'mlp':
                outputs = model(batch[0], batch[1], batch[2])
            else:
                inputs = {'input_ids':      batch[0],
                          'attention_mask': batch[1],
                          'labels':         batch[3]}
                if args.model_type != 'distilbert':
                    inputs['token_type_ids'] = batch[2] if args.model_type in ['bert', 'xlnet'] else None  # XLM, DistilBERT and RoBERTa don't use segment_ids
                outputs = model(**inputs)
            loss = outputs[0]
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            loss.backward()
            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_step += 1
                if WANDB:
                    wandb.log({'epoch': epoch,
                               'lr': scheduler.get_last_lr()[0],
                               'loss': loss})

            if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                # if args.evaluate_during_training:
                #     results = evaluate(args, dev_data, model, tokenizer)
                #     for key, value in results.items():
                #         tb_writer.add_scalar('eval_{}'.format(key), value, global_step)

                avg_loss = (tr_loss - logging_loss)/ args.logging_steps
                writer.add_scalar('lr', scheduler.get_last_lr()[0], global_step)
                writer.add_scalar('loss', avg_loss, global_step)
                logging_loss = tr_loss

    writer.close()
    return global_step, tr_loss / global_step



def evaluate(args, dev_or_test_data, model, tokenizer):
    if args.model_type == 'mlp':
        collate_fn = collate_for_mlp
    else:
        collate_fn = get_collate_for_transformer(tokenizer.pad_token_id)
    data_loader = torch.utils.data.DataLoader(dev_or_test_data,
                                              collate_fn=collate_fn,
                                              num_workers=args.num_workers,
                                              batch_size=args.test_batch_size,
                                              pin_memory=('cuda' in str(args.device)),
                                              shuffle=False)
    all_logits = []
    all_targets = []
    nb_eval_steps, eval_loss = 0, 0.0
    for batch in tqdm(data_loader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            if args.model_type == 'mlp':
                # batch consist of (flat_inputs, lenghts, labels)
                outputs = model(batch[0], batch[1], batch[2])
                all_targets.append(batch[2].detach().cpu())
            else:
                inputs = {'input_ids':      batch[0],
                          'attention_mask': batch[1],
                          'labels':         batch[3]}
                if args.model_type != 'distilbert':
                    inputs['token_type_ids'] = batch[2] if args.model_type in ['bert', 'xlnet'] else None  # XLM, DistilBERT and RoBERTa don't use segment_ids
                outputs = model(**inputs)
                all_targets.append(inputs['labels'].detach().cpu())
        nb_eval_steps += 1
        # outputs [:2] should hold loss, logits
        loss, logits = outputs[:2]
        eval_loss += loss.mean().item()
        all_logits.append(logits.detach().cpu())

    logits = torch.cat(all_logits).numpy()
    targets = torch.cat(all_targets).numpy()
    eval_loss /= nb_eval_steps
    preds = np.argmax(logits, axis=1)
    acc = (preds == targets).sum() / targets.size

    f1_micro = f1_score(targets, preds, average='micro')
    f1_macro = f1_score(targets, preds, average='macro', zero_division=1)

    if WANDB:
        wandb.log({"test/acc": acc, "test/loss": eval_loss,
                   "test/f1_micro": f1_micro,
                   "test/f1_macro": f1_macro})
    return acc, eval_loss


def run_xy_model(args):
    print("Loading data...")

    if args.model_type == "mlp" and args.model_name_or_path is not None:
        print("Assuming to use word embeddings as both model_type=mlp and model_name_or_path are given")
        print("Using word embeddings -> forcing wordlevel tokenizer")
        vocab, embedding = load_word_vectors(args.model_name_or_path, unk_token="[UNK]")
        tokenizer = build_tokenizer_for_word_embeddings(vocab)
    else:
        tokenizer_name = args.tokenizer_name if args.tokenizer_name else args.model_name_or_path
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        embedding = None
    print("Using tokenizer:", tokenizer)

    do_truncate = not (args.stats_and_exit or args.model_type == 'mlp')
    if args.stats_and_exit:
        # We only compute dataset stats including length, so NOT truncate
        max_length = None
    elif args.model_type == 'mlp':
        max_length = None
    else:
        max_length = 512 # should hold for all used transformer models?

    enc_docs, enc_labels, train_mask, test_mask, label2index = load_data(args.dataset,
                                                                         tokenizer,
                                                                         max_length=max_length,
                                                                         construct_textgraph=False,
                                                                         n_jobs=args.num_workers)
    print("Done")

    lens = np.array([len(doc) for doc in enc_docs])
    print("Min/max document length:", (lens.min(), lens.max()))
    print("Mean document length: {:.4f} ({:.4f})".format(lens.mean(), lens.std()))
    assert len(enc_docs) == len(enc_labels) == train_mask.size(0) == test_mask.size(0)
    enc_docs_arr, enc_labels_arr = np.array(enc_docs, dtype='object'), np.array(enc_labels)


    train_data = list(zip(enc_docs_arr[train_mask], enc_labels_arr[train_mask]))
    test_data = list(zip(enc_docs_arr[test_mask], enc_labels_arr[test_mask]))

    print("N", len(enc_docs))
    print("N train", len(train_data))
    print("N test", len(test_data))
    print("N classes", len(label2index))


    if args.stats_and_exit:
        print("Warning: length stats depend on tokenizer and max_length of model, chose MLP to avoid trimming before computing stats.")
        exit(0)

    if args.model_type != 'mlp':
        config_class, model_class, __ = MODEL_CLASSES[args.model_type]
        print("Loading", args.model_type)
        print("Loading config")
        config = config_class.from_pretrained(args.model_name_or_path,
                                              num_labels=len(label2index),
                                              cache_dir=CACHE_DIR)
        print(config)
        print("Loading model")
        model = model_class.from_pretrained(args.model_name_or_path,
                                            from_tf=bool('.ckpt' in args.model_name_or_path),
                                            config=config,
                                            cache_dir=CACHE_DIR)
    else:
        print("Initializing MLP")

        if embedding is not None:
            # Vocab size given by embedding
            vocab_size = None 
        else:
            vocab_size = tokenizer.vocab_size

        if args.bow_aggregation == 'tfidf':
            print("Using IDF")
            idf = inverse_document_frequency(enc_docs_arr[train_mask], tokenizer.vocab_size).to(args.device)
        else:
            idf = None

        model = MLP(vocab_size, len(label2index),
                    num_hidden_layers=args.mlp_num_layers,
                    hidden_size=args.mlp_hidden_size,
                    embedding_dropout=args.mlp_embedding_dropout,
                    dropout=args.mlp_dropout,
                    mode=args.bow_aggregation,
                    pretrained_embedding=embedding,
                    idf=idf,
                    freeze=args.freeze_embedding)

    model.to(args.device)


    if WANDB:
        wandb.watch(model, log_freq=args.logging_steps)


    train(args, train_data, model, tokenizer)
    acc, eval_loss = evaluate(args, test_data, model, tokenizer)
    print(f"[{args.dataset}] Test accuracy: {acc:.4f}, Eval loss: {eval_loss}")
    return acc


def run_axy_model(args):
    raise NotImplementedError("We did not run TextGCN ourselves. This implementation is deprecated. Please use --model_type mlp or distilbert")
    print("Loading data...")
    tokenizer_name = args.tokenizer_name if args.tokenizer_name else args.model_name_or_path

    do_truncate = not (args.stats_and_exit or args.model_type == 'mlp')
    data = load_data(args.dataset,
                     tokenizer_name,
                     truncate=do_truncate,
                     construct_textgraph=True,
                     n_jobs=args.num_workers)
    print("Done")
    print("N", data.x.size(0))
    print("N words", data.word_mask.sum().item())
    print("N documents", (data.train_mask | data.test_mask).sum().item())
    print("N train", data.train_mask.sum().item())

    print("Edge index size", data.edge_index.size(), data.edge_index.dtype)
    print("Edge attr size", data.edge_attr.size(), data.edge_attr.dtype)
    print("Is undirected", tg.utils.is_undirected(data.edge_index))
    num_labels = len(data.label2index)
    print("N classes", num_labels)
    print("Asserting that train and test masks are disjunct")
    assert not (data.train_mask & data.test_mask).any()
    print("Okay.")
    gnn_hidden = 200
    gnn_dropout = 0.1
    gnn_activation = F.relu

    data.edge_attr = data.edge_attr.squeeze().float()

    if args.model_type == 'textgcn':
        # Text GCN uses identity matrix features
        onehot_features = torch.eye(data.x.size(0), data.x.size(0))
        del data.x
        data.x = onehot_features
        model = GCN(data.x.size(1), gnn_hidden, num_labels,
                    dropout=gnn_dropout, activation=gnn_activation)

    else:
        config_class, __, model_class = MODEL_CLASSES[args.model_type]
        config = config_class.from_pretrained(args.model_name_or_path,
                                              cache_dir=CACHE_DIR)
        print(config)
        transformer = model_class.from_pretrained(args.model_name_or_path,
                                                  from_tf=bool('.ckpt' in args.model_name_or_path),
                                                  config=config,
                                                  cache_dir=CACHE_DIR)
        model = TransformerForNodeClassification(transformer, config.dim,
                                                 gnn_hidden, num_labels,
                                                 activation=gnn_activation,
                                                 dropout=gnn_dropout)

    model.to(args.device)
    data.to(args.device)

    if WANDB:
        wandb.watch(model, log_freq=100)

    # TRAIN #
    t_total = args.epochs * args.gradient_accumulation_steps

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon)
    # scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)
    writer = SummaryWriter()

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    train_iterator = trange(args.epochs, desc="Epoch")
    for step in train_iterator:
        model.train()
        model.zero_grad()
        if args.model_type == 'textgcn':
            outputs = model(data.x,
                            data.edge_index,
                            data.edge_attr,
                            labels=data.y,
                            mask=data.train_mask)
        else:
            raise NotImplementedError("TODO")

        loss = outputs[0]
        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps
        loss.backward()
        tr_loss += loss.item()
        if (step + 1) % args.gradient_accumulation_steps == 0:
            optimizer.step()
            scheduler.step()
            model.zero_grad()
            global_step += 1
        if args.logging_steps > 0 and global_step % args.logging_steps == 0:
            # if args.evaluate_during_training:
            #     acc, eval_loss = evaluate(args, dev_data, model, tokenizer)
            #     tb_writer.add_scalar('eval_acc', acc, global_step)
            #     tb_writer.add_scalar('eval_loss', eval_loss, global_step)
            #     if WANDB:
            #         wandb.log({"eval/acc": acc, "eval/loss": eval_loss})
            current_lr = scheduler.get_last_lr()[0]
            avg_loss = (tr_loss - logging_loss)/ args.logging_steps

            writer.add_scalar('lr', current_lr, global_step)
            writer.add_scalar('loss', avg_loss, global_step)

            if WANDB:
                wandb.log({'lr': current_lr,
                            'loss': avg_loss})
            logging_loss = tr_loss

    writer.close()

    # EVALUATE #
    all_logits = []
    all_targets = []
    nb_eval_steps, eval_loss = 0, 0.0
    model.eval()
    with torch.no_grad():
        if args.model_type == 'textgcn':
            outputs = model(data.x,
                            data.edge_index,
                            data.edge_attr,
                            labels=data.y,
                            mask=data.test_mask)
        else:
            raise NotImplementedError("TODO")

        eval_loss, logits = outputs[:2]
    preds = np.argmax(logits.detach().cpu().numpy(), axis=1)
    targets = data.y[data.test_mask].numpy()
    print("preds", preds.shape, targets.dtype)
    print("targets", targets.shape, targets.dtype)
    acc = (preds == targets).sum() / targets.size

    print(f"[{args.dataset}] Test accuracy: {acc:.4f}, Eval loss: {eval_loss}")
    writer.add_hparams(
        hparam_dict={ 'learning_rate': args.learning_rate,
                      'gradient_accumulation_steps': args.gradient_accumulation_steps,
                      'epochs': args.epochs },
        metric_dict={ 'test_accuracy' : acc }
    )
    return acc


def main():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('dataset', choices=VALID_DATASETS)
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type: either 'mlp' or 'distilbert'",
                        choices=["mlp", "distilbert"])
    parser.add_argument("--model_name_or_path", default=None, type=str,
                        help="Optional path to word embedding with model type 'mlp' OR huggingface shortcut name such as distilbert-base-uncased for model type 'distilbert'")
    parser.add_argument("--results_file", default=None,
                        help="Store results to this results file")

    ## Training config
    parser.add_argument("--epochs", type=int, default=20,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for training")
    parser.add_argument("--test_batch_size", type=int, default=None,
                        help="Batch size for testing (defaults to train batch size)")
    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")

    parser.add_argument('--unfreeze_embedding', dest="freeze_embedding", default=True,
            action='store_false', help="Allow updating pretrained embeddings")

    ## Training Hyperparameters
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")

    ## Other parameters
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--num_workers", default=4, type=int,
                        help="Number of workers")

    parser.add_argument("--stats_and_exit", default=False,
                        action='store_true',
                        help="Print dataset stats and exit.")

    # MLP Params
    parser.add_argument("--mlp_num_layers", default=1, type=int, help="Number of hidden layers within MLP")
    parser.add_argument("--mlp_hidden_size", default=1024, type=int, help="Hidden dimension for MLP")
    parser.add_argument("--bow_aggregation", default="mean", choices=["mean", "sum", "tfidf"],
            help="Aggregation for bag-of-words models (such as MLP)")
    parser.add_argument("--mlp_embedding_dropout", default=0.5, type=float, help="Dropout for embedding / first hidden layer ")
    parser.add_argument("--mlp_dropout", default=0.5, type=float, help="Dropout for all subsequent layers")

    parser.add_argument("--comment", help="Some comment for the experiment")
    args = parser.parse_args()

    if args.model_type in ['mlp', 'textgcn']:
        assert args.tokenizer_name or args.model_name_or_path, "Please supply tokenizer for MLP via --tokenizer_name or provide an embedding via --model_name_or_path"
    else:
        assert args.model_name_or_path, f"Please supply --model_name_or_path for {args.model_type}"

    args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    args.test_batch_size = args.batch_size if args.test_batch_size is None else args.test_batch_size

    if WANDB:
        wandb.init(project="text-clf")
        wandb.config.update(args)

    acc = {
        'mlp': run_xy_model,
        'bert': run_xy_model,
        'distilbert': run_xy_model,
        'roberta': run_xy_model,
        'xlnet': run_xy_model,
        'textgcn': run_axy_model,
    }[args.model_type](args)
    if args.results_file:
        with open(args.results_file, 'a', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow([args.model_type,args.dataset,acc])


if __name__ == '__main__':
    main()
