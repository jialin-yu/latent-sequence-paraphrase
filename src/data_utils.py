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


def process_quora(file_path_read, use_spacy=True):   
    print(f'Read quora data from path: {file_path_read}...')
    
    with open(file_path_read, 'r') as f:
        lines = f.readlines()[1:] # ignore header line
    
    sentence_pairs = []
    for l in tqdm(lines):
        if len(l.split('\t') ) != 6: # ignore error format
            continue
        q1, q2, is_duplicate = l.split('\t')[3:]         
        if int(is_duplicate) == 1:
            sentence_pairs.append((tokenizer(q1, use_spacy), tokenizer(q2, use_spacy), pseudo_tokenizer(q1, use_spacy)))

    print(f'Read {len(sentence_pairs)} pairs from original {len(lines)} pairs.')
    return sentence_pairs  

def process_mscoco(file_path_read, use_spacy=True):   
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
        l = shuffle(l, random_state=1234)
        q1, q2, q3, q4, q5 = l
        sentence_pairs.append((tokenizer(q1, use_spacy), tokenizer(q2, use_spacy),tokenizer(q3, use_spacy), tokenizer(q4, use_spacy),tokenizer(q5, use_spacy), pseudo_tokenizer(q1, use_spacy)))    

    print(f'Read {len(sentence_pairs)} pairs from original {len(sentence_sets)} sets.')
    return sentence_pairs

#################################################
############## Calculate Statistics #############
#################################################

def calculate_stats(sentence_sets):
    
    tmp = []
    for sets in tqdm(sentence_sets):  
        tmp.append(len(sets[0]))
        tmp.append(len(sets[1]))

    np_arr = np.array(tmp)
    print(f'Mean: {np.ceil(np.mean(np_arr))}; STD: {np.ceil(np.std(np_arr))}; Min: {np.ceil(np.min(np_arr))} and Max: {np.ceil(np.max(np_arr))}')

#################################################
############## Build Vocabulary     #############
#################################################

def create_vocab(sentence_sets, min_freq=1, max_size=None):
    MIN_FREQUENT = min_freq
    print(f'Creating vocab object ...')
    
    counter = Counter()
    for sets in tqdm(sentence_sets):
        if len(sets) == 3:           
            counter.update(sets[0])
            counter.update(sets[1])
        else:
            counter.update(sets[0])
            counter.update(sets[1])
            counter.update(sets[2])
            counter.update(sets[3])
            counter.update(sets[4])

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

    tr_v_temp = []
    t_temp = []

    for sets in tqdm(train_and_valid): 
        tr_v_temp.append((token_to_index(sets[0][:cutoff], vocab), token_to_index(sets[1][:cutoff], vocab), token_to_index(sets[-1][:cutoff], vocab)))
    for sets in tqdm(test):  
        t_temp.append((token_to_index(sets[0][:cutoff], vocab), token_to_index(sets[1][:cutoff], vocab), token_to_index(sets[-1][:cutoff], vocab)))
    
    return tr_v_temp, t_temp

#################################################
################  Calculate Bound   #############
#################################################

bleu_metric = load_metric('bleu')
rouge_metric = load_metric('rouge')
bertscore_metric = load_metric("bertscore")

def calculate_bound(pred_set, reference_set, bleu=False, rouge=False, inference=False):
    '''
    pred_set and reference_set in tokenized format
    pred_set = [[tokenized_1], [tokenized_2], ...]
    reference_set = [([tokenized_1_1], [tokenized_1_2]), ([tokenized_2_1], [tokenized_2_2]), ...]
    '''

    shuffle_pred_set= shuffle(pred_set, random_state=1234)

    if(bleu):

        print(f'{"-"*20} Calculate BLEU score {"-"*20}')
        pred = pred_set
        refer = [[s_ for s_ in s[1:-1]] for s in reference_set]

        bleu = bleu_metric.compute(predictions=pred, references=refer)

        pred = shuffle_pred_set

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
        pred = [stringify(s) for s in pred_set]
        refer = [stringify(s[1]) for s in reference_set]

        rouge = rouge_metric.compute(predictions=pred, references=refer)

        pred = [stringify(s) for s in shuffle_pred_set]
        
        rouge_ = rouge_metric.compute(predictions=pred, references=refer)
        
        if (inference):
            print(f'ROUGE-1 score is {rouge["rouge1"].mid.fmeasure}, ROUGE-2 score is {rouge["rouge2"].mid.fmeasure}, and ROUGE-L score is {rouge["rougeL"].mid.fmeasure}.')
        else:
            print(f'{"-"*20} Ground truth upper bound {"-"*20}')
            print(f'ROUGE-1 score is {rouge["rouge1"].mid.fmeasure}, ROUGE-2 score is {rouge["rouge2"].mid.fmeasure}, and ROUGE-L score is {rouge["rougeL"].mid.fmeasure}.')
            print(f'{"-"*20} Random selection lower bound {"-"*20}')
            print(f'ROUGE-1 score is {rouge_["rouge1"].mid.fmeasure}, ROUGE-2 score is {rouge_["rouge2"].mid.fmeasure}, and ROUGE-L score is {rouge_["rougeL"].mid.fmeasure}.')

    a = True
    if (a):

        print(f'{"-"*20} Calculate BERT score {"-"*20}')
        pred = [stringify(s) for s in pred_set]
        refer = [stringify(s[1]) for s in reference_set]

        bert = bertscore_metric.compute(predictions=pred, references=refer, lang="en")

        pred = [stringify(s) for s in shuffle_pred_set]
        
        bert_ = bertscore_metric.compute(predictions=pred, references=refer, lang="en")
        
        if (inference):
            print(f'BERT score precision is {bert["precision"]}, recall is {bert["recall"]}, and F1 is {bert["f1"]}.')
        else:
            print(f'{"-"*20} Ground truth upper bound {"-"*20}')
            print(f'BERT score precision is {bert["precision"]}, recall is {bert["recall"]}, and F1 is {bert["f1"]}.')
            print(f'{"-"*20} Random selection lower bound {"-"*20}')
            print(f'BERT score precision is {bert_["precision"]}, recall is {bert_["recall"]}, and F1 is {bert_["f1"]}.')