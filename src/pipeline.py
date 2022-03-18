import spacy
import random
from sklearn.utils import shuffle

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

def pseudo_tokenizer(string):
    return shuffle(tokenizer(string), random_state=1234)

def remove_bos_eos(idx_list, bos_id, eos_id):
    clear_idx = []
    for idx in idx_list:
        if idx == eos_id:
            return clear_idx
        else:
            if idx == bos_id:
                continue
            else:
                clear_idx.append(idx)
    
    return clear_idx

