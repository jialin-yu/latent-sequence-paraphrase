from datasets import temp_seed
import spacy
import random

spacy_pipeline = spacy.load('en_core_web_sm')


def tokenizer(string):
    string_ = string.strip()
    return [token.text for token in spacy_pipeline.tokenizer(string_.lower())]

def stringify(token):
    return ' '.join(token)

def token_to_index(token, vocab_object):
    return [vocab_object[tok] for tok in token]

def index_to_token(index, vocab_object):
    return [vocab_object.get_itos()[ind] for ind in index]

def get_pseudo(string):
    tmp_str_token = string
    random.Random(1234).shuffle(tokenizer(tmp_str_token))
    return tmp_str_token

