""" Tokenizers for usage with word embeddings """
from tokenizers import Tokenizer, normalizers
from tokenizers.models import WordLevel
from tokenizers.normalizers import Lowercase, NFD, StripAccents
from tokenizers.pre_tokenizers import Whitespace


def build_tokenizer_for_word_embeddings(vocab):
    """ Build a word level tokenizer for word embeddings """
    model = WordLevel(vocab, "[UNK]")
    tokenizer = Tokenizer(model)
    tokenizer.normalizer = normalizers.Sequence([NFD(), Lowercase(), StripAccents()])
    tokenizer.pre_tokenizer = Whitespace()
    return tokenizer
