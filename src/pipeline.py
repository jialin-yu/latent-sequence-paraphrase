from transformers import BertTokenizer

bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# set bos and eos token as cls and sep
bert_tokenizer.bos_token = bert_tokenizer.cls_token
bert_tokenizer.eos_token = bert_tokenizer.sep_token


def tokenizer(string):
    return bert_tokenizer.tokenize(string)

def stringify(token):
    return bert_tokenizer.convert_tokens_to_string(token)

def token_to_index(token):
    return bert_tokenizer.convert_tokens_to_ids(token)

def index_to_token(index):
    return bert_tokenizer.convert_ids_to_tokens(index)

def remove_bos_eos(index):
    clear_idx = []
    for idx in index:
        if idx == bert_tokenizer.eos_token_id:
            return remove_pad(clear_idx)
        else:
            if idx == bert_tokenizer.bos_token_id:
                continue
            else:
                clear_idx.append(idx)
    
    return remove_pad(clear_idx)

def remove_pad(index):
    clear_idx = []
    for idx in index:
        if idx == bert_tokenizer.pad_token_id:
            continue
        else:
            clear_idx.append(idx)
    return clear_idx
