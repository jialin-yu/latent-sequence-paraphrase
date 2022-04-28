from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
tokenizer.bos_token = tokenizer.cls_token
tokenizer.eos_token = tokenizer.sep_token


def tokenize(string):
    return tokenizer.tokenize(string)

def stringify(token):
    return tokenizer.convert_tokens_to_string(token)

def token_to_index(token):
    return tokenizer.convert_tokens_to_ids(token)

def index_to_token(index):
    return tokenizer.convert_ids_to_tokens(index)

def decode_index_to_string(index):
    clear_idx = clear_index(index)
    return tokenizer.decode(clear_idx, skip_special_tokens=True)

def index_to_string(index):
    return stringify(index_to_token(index))
    
def clear_index(index):
    clear_idx = []
    for idx in index:
        if idx == tokenizer.eos_token_id:
            clear_idx.append(idx)
            return clear_idx
        else:
            clear_idx.append(idx)
    return clear_idx
