import spacy

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
