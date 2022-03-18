'''
Data pre-/post- processing functions

Contains two datasets:
* MSCOCO (Lin et. al. 2014)] (http://cocodataset.org/)
* Quora (https://quoradata.quora.com/First-Quora-Dataset-Release-Question-Pairs)

28TH FEB 2022
BY Jialin Yu
'''

import json
import random
from datasets import load_metric
import numpy as np
from pipeline import tokenizer, token_to_index, stringify, pseudo_tokenizer
from tqdm import tqdm

from torchtext.vocab import vocab
from collections import Counter, OrderedDict
from sklearn.utils import shuffle


#################################################
############## Process Raw Data   ###############
#################################################


def process_quora(file_path_read):   
    print(f'Read quora data from path: {file_path_read}...')
    
    with open(file_path_read, 'r') as f:
        lines = f.readlines()[1:] # ignore header line
    
    sentence_pairs = []
    for l in tqdm(lines):
        if len(l.split('\t') ) != 6: # ignore error format
            continue
        q1, q2, is_duplicate = l.split('\t')[3:]         
        if int(is_duplicate) == 1:
            sentence_pairs.append((tokenizer(q1), tokenizer(q2), pseudo_tokenizer(q1)))

    print(f'Read {len(sentence_pairs)} pairs from original {len(lines)} pairs.')
    return sentence_pairs  

def process_mscoco(file_path_read):   
    print(f'Read mscoco data from path: {file_path_read}...')
    
    with open(file_path_read, 'r') as f:
        data = json.load(f)
    
    # aggregate all sentences of the same images
    image_idx = set([d["image_id"] for d in data["annotations"]])
    paraphrases = {}
    for im in image_idx: 
        paraphrases[im] = []
    for d in tqdm(data["annotations"]):
        im = d["image_id"]
        sent = d["caption"]
        paraphrases[im].append(sent)
    sentence_sets = [paraphrases[im] for im in paraphrases]

    sentence_pairs = []
    for l in tqdm(sentence_sets):
        if len(l) != 5: # ignore error format
            continue
        q1, q2, q3, q4, q5 = l
        sentence_pairs.append((tokenizer(q1), tokenizer(q2), pseudo_tokenizer(q1)))
        sentence_pairs.append((tokenizer(q3), tokenizer(q4), pseudo_tokenizer(q3)))    

    print(f'Read {len(sentence_pairs)} pairs from original {len(sentence_sets)} sets ({2*len(sentence_sets)} pairs).')
    return sentence_pairs

#################################################
############## Calculate Statistics #############
#################################################

def calculate_stats(sentence_pair):
    tmp = []
    for (q1, q2, _) in tqdm(sentence_pair):
        tmp.append(len(q1))
        tmp.append(len(q2))
    np_arr = np.array(tmp)
    print(f'Mean: {np.ceil(np.mean(np_arr))}; STD: {np.ceil(np.std(np_arr))}; Min: {np.ceil(np.min(np_arr))} and Max: {np.ceil(np.max(np_arr))}')

#################################################
############## Build Vocabulary     #############
#################################################

def create_vocab(sentence_pair, min_freq=1, max_size=None):
    MIN_FREQUENT = min_freq
    print(f'Creating vocab object ...')
    counter = Counter()
    for (q1, q2, _) in tqdm(sentence_pair):
        counter.update(q1)
        counter.update(q2)
    
    sorted_by_freq_tuples = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    if max_size == None:
        MAX_SIZE = len(sorted_by_freq_tuples)
    else:
        MAX_SIZE = max_size
    sorted_by_freq_tuples = sorted_by_freq_tuples[:MAX_SIZE]
    ordered_dict = OrderedDict(sorted_by_freq_tuples)
    VOCAB = vocab(ordered_dict, MIN_FREQUENT)
    print(f'Vocab object created with size {len(VOCAB)}.')
    return VOCAB

def append_special_tokens(vocab_object, sp_tokens, unk_id):
    print(f'Assigning special tokens of size {len(sp_tokens)}.')
    for index, token in enumerate(tqdm(sp_tokens)): 
        vocab_object.insert_token(token, index)      
    vocab_object.set_default_index(unk_id)
    print(f'Set default token as {sp_tokens[unk_id]}.') 
    print(f'Vocabulary size is now {len(vocab_object)}.')
    return vocab_object

#################################################
################   Normalise Data   #############
#################################################

def normalise(train_and_valid, test, vocab, cutoff):
    print(f'Normalising data...')

    train_valid_temp = []
    for q1, q2, q1_ in tqdm(train_and_valid):
        train_valid_temp.append((token_to_index(q1[:cutoff], vocab), token_to_index(q2[:cutoff], vocab), token_to_index(q1_[:cutoff], vocab)))
    
    test_temp = []
    for q1, q2, q1_ in tqdm(test):
        test_temp.append((token_to_index(q1[:cutoff], vocab), token_to_index(q2[:cutoff], vocab), token_to_index(q1_[:cutoff], vocab)))
    return train_valid_temp, test_temp

#################################################
################  Calculate Bound   #############
#################################################

bleu_metric = load_metric('bleu')
rouge_metric = load_metric('rouge')

def calculate_bound(tokenized_test_pairs, bleu=False, rouge=False, inference=False):

    shuffle_test_pairs = shuffle(tokenized_test_pairs, random_state=1234)

    if(bleu):

        print(f'{"-"*20} Calculate BLEU score {"-"*20}')
        pred = [s[0] for s in tokenized_test_pairs]
        refer = [[s[1]] for s in tokenized_test_pairs]

        bleu = bleu_metric.compute(predictions=pred, references=refer)

        pred = [s[0] for s in shuffle_test_pairs]
        refer = [[s[1]] for s in tokenized_test_pairs]

        bleu_ = bleu_metric.compute(predictions=pred, references=refer)
        
        if (inference):
            print(f'BLEU score is {bleu["bleu"]} and precisions are {bleu["precisions"]}.')
        else:
            print(f'{"-"*20} Ground truth upper bound {"-"*20}')
            print(f'BLEU score is {bleu["bleu"]} and precisions are {bleu["precisions"]}.')
            print(f'{"-"*20} Random selection lower bound {"-"*20}')
            print(f'BLEU score is {bleu_["bleu"]} and precisions are {bleu_["precisions"]}.')
    
    if (rouge):

        print(f'{"-"*20} Calculate ROUGE score {"-"*20}')

        pred = [stringify(s[0]) for s in tokenized_test_pairs]
        refer = [stringify(s[1]) for s in tokenized_test_pairs]

        rouge = rouge_metric.compute(predictions=pred, references=refer)

        pred = [stringify(s[0]) for s in shuffle_test_pairs]
        refer = [stringify(s[1]) for s in tokenized_test_pairs]
        
        rouge_ = rouge_metric.compute(predictions=pred, references=refer)
        
        if (inference):
            print(f'ROUGE-1 score is {rouge["rouge1"].mid.fmeasure}, ROUGE-2 score is {rouge["rouge2"].mid.fmeasure}, and ROUGE-L score is {rouge["rougeL"].mid.fmeasure}.')
        else:
            print(f'{"-"*20} Ground truth upper bound {"-"*20}')
            print(f'ROUGE-1 score is {rouge["rouge1"].mid.fmeasure}, ROUGE-2 score is {rouge["rouge2"].mid.fmeasure}, and ROUGE-L score is {rouge["rougeL"].mid.fmeasure}.')
            print(f'{"-"*20} Random selection lower bound {"-"*20}')
            print(f'ROUGE-1 score is {rouge_["rouge1"].mid.fmeasure}, ROUGE-2 score is {rouge_["rouge2"].mid.fmeasure}, and ROUGE-L score is {rouge_["rougeL"].mid.fmeasure}.')
