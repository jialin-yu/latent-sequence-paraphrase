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
from git import reference
import numpy as np
from pipeline import tokenize, token_to_index, stringify
from tqdm import tqdm

from torchtext.vocab import vocab
from collections import Counter, OrderedDict
from sklearn.utils import shuffle
from nltk.translate.bleu_score import corpus_bleu


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
            sentence_pairs.append((tokenize(q1), tokenize(q2)))

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
        l = shuffle(l, random_state=1234)
        q1, q2, q3, q4, q5 = l
        sentence_pairs.append((tokenize(q1), tokenize(q2),tokenize(q3), tokenize(q4), tokenize(q5)))    

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
################   Normalise Data   #############
#################################################

def normalise(train_and_valid, test, cutoff):
    print(f'Normalising data...')

    tr_v_temp = []
    t_temp = []

    for sets in tqdm(train_and_valid): 
        # tr_v_temp.append((token_to_index(sets[0][:cutoff]), token_to_index(sets[1][:cutoff])))
        tr_v_temp.append([token_to_index(item[:cutoff]) for item in sets[:2]])
    for sets in tqdm(test):  
        # t_temp.append((token_to_index(sets[0][:cutoff]), token_to_index(sets[1][:cutoff])))
        t_temp.append([token_to_index(item[:cutoff]) for item in sets[:2]])
    
    return tr_v_temp, t_temp

#################################################
################  Calculate Bound   #############
#################################################

bleu_metric = load_metric('bleu')
# from nltk.translate.bleu_score import corpus_bleu
rouge_metric = load_metric('rouge')
# bertscore_metric = load_metric("bertscore")

def calculate_bleu(preds, trg_refers):
    '''
    Return bleu score based on preds and refers
    '''
    B_1 = corpus_bleu(trg_refers, preds, weights=(1, 0, 0, 0))
    B_2 = corpus_bleu(trg_refers, preds, weights=(0.5, 0.5, 0, 0))
    B_3 = corpus_bleu(trg_refers, preds, weights=(0.33, 0.33, 0.34, 0))
    B_4 = corpus_bleu(trg_refers, preds, weights=(0.25, 0.25, 0.25, 0.25))
    B = corpus_bleu(trg_refers, preds)

    return [B_1, B_2, B_3, B_4, B]


def calculate_i_bleu(preds, src_refers, trg_refers, alpha=0.8):
    '''
    default use alpha = 0.8; https://arxiv.org/pdf/2203.03463v2.pdf; 5.2 Paraphrase Generation
    https://aclanthology.org/2021.findings-acl.50.pdf to reproduce the baseline result
    i-bleu = alpha*bleu(output, trg_refer ) + (1-alpha)*bleu(output, src_refer)
    '''
    return alpha * corpus_bleu(trg_refers, preds) - (1-alpha) * corpus_bleu(src_refers, preds)


def calculate_self_bleu(preds, src_refers):
    '''
    Return i-bleu score based on preds and refers
    '''
    return corpus_bleu(src_refers, preds)


def calculate_bound(pred_set, reference_set, bleu=False, rouge=False, inference=False):
    '''
    pred_set and reference_set in tokenized format
    pred_set = [[tokenized_1], [tokenized_2], ...]
    reference_set = [([tokenized_1_1], [tokenized_1_2]), ([tokenized_2_1], [tokenized_2_2]), ...]
    '''

    shuffle_pred_set= shuffle(pred_set, random_state=1234)

    if(bleu):

        print(f'{"-"*20} Calculate BLEU score {"-"*20}')
        # pre = [[s] for s in pred_set]
        src_refer = [[s[0]] for s in reference_set]
        trg_refer = [[s_ for s_ in s[1:]] for s in reference_set]
        # refer = [[s[1:-1]] for s in reference_set]
        
        upper_pred = pred_set

        upper_bleu = calculate_bleu(upper_pred, trg_refer)
        upper_i_bleu = calculate_i_bleu(upper_pred, src_refer, trg_refer)
        upper_self_bleu =  calculate_self_bleu(upper_pred, src_refer)

        lower_pred = shuffle_pred_set

        lower_bleu = calculate_bleu(lower_pred, trg_refer)
        lower_i_bleu = calculate_i_bleu(lower_pred, src_refer, trg_refer)
        lower_self_bleu =  calculate_self_bleu(lower_pred, src_refer)

        if (inference):
            print(f'BLEU-1: {upper_bleu[0]}; BLEU-2: {upper_bleu[1]}; BLEU-3: {upper_bleu[2]}; BLEU-4: {upper_bleu[3]}; and BLEU {upper_bleu[4]} .')
            print(f'self-BLEU: {upper_self_bleu} and i-BLEU: {upper_i_bleu} .')
        else:
            print(f'{"-"*20} Ground truth upper bound {"-"*20}')
            print(f'BLEU-1: {upper_bleu[0]}; BLEU-2: {upper_bleu[1]}; BLEU-3: {upper_bleu[2]}; BLEU-4: {upper_bleu[3]}; and BLEU {upper_bleu[4]} .')
            print(f'self-BLEU: {upper_self_bleu} and i-BLEU: {upper_i_bleu} .')
            print(f'{"-"*20} Random selection lower bound {"-"*20}')
            print(f'BLEU-1: {lower_bleu[0]}; BLEU-2: {lower_bleu[1]}; BLEU-3: {lower_bleu[2]}; BLEU-4: {lower_bleu[3]}; and BLEU {lower_bleu[4]} .')
            print(f'self-BLEU: {lower_self_bleu} and i-BLEU: {lower_i_bleu} .')
            print(f'{"-"*40}')
    
    if (rouge):

        print(f'{"-"*20} Calculate ROUGE score {"-"*20}')
        upper_pred = [stringify(s) for s in pred_set]
        refer = [stringify(s[1]) for s in reference_set]

        upper_rouge = rouge_metric.compute(predictions=upper_pred, references=refer)

        lower_pred = [stringify(s) for s in shuffle_pred_set]
        
        lower_rouge = rouge_metric.compute(predictions=lower_pred, references=refer)
        
        if (inference):
            print(f'ROUGE-1 score is {upper_rouge["rouge1"].mid.fmeasure}, ROUGE-2 score is {upper_rouge["rouge2"].mid.fmeasure}, and ROUGE-L score is {upper_rouge["rougeL"].mid.fmeasure}.')
        else:
            print(f'{"-"*20} Ground truth upper bound {"-"*20}')
            print(f'ROUGE-1 score is {upper_rouge["rouge1"].mid.fmeasure}, ROUGE-2 score is {upper_rouge["rouge2"].mid.fmeasure}, and ROUGE-L score is {upper_rouge["rougeL"].mid.fmeasure}.')
            print(f'{"-"*20} Random selection lower bound {"-"*20}')
            print(f'ROUGE-1 score is {lower_rouge["rouge1"].mid.fmeasure}, ROUGE-2 score is {lower_rouge["rouge2"].mid.fmeasure}, and ROUGE-L score is {lower_rouge["rougeL"].mid.fmeasure}.')


def gumble_search(pred_set, pred_samples, reference_set):
    ''''
    search with pre-trained model and return the best similarity with the source text, in tokenised form
    '''
    bertscore = load_metric("bertscore")

    final_search = []

    src_string = [stringify(s[0]) for s in reference_set]

    assert len(pred_set) == len(pred_samples)
    for index, _ in enumerate(pred_set):
        search_text = pred_samples[index] + [pred_set[index]] # [[tokenized], [tokenized]]
        # print(pred_samples[index])
        # print(pred_set[index])
        # print(search_text)
        search_text_string = [stringify(s) for s in search_text] # [s1, s2, s3]
        source_text = src_string[index]
        res = [bertscore.compute(predictions=[refer], references=[source_text], lang="en")['f1'][0] for refer in search_text_string]
        # print(res)
        final_search.append(search_text[res.index(max(res))])
    
    return final_search


    # a = False
    # if (a):

    #     print(f'{"-"*20} Calculate BERT score {"-"*20}')
    #     pred = [stringify(s) for s in pred_set]
    #     refer = [stringify(s[1]) for s in reference_set]

    #     bert = bertscore_metric.compute(predictions=pred, references=refer, lang="en")

    #     pred = [stringify(s) for s in shuffle_pred_set]
        
    #     bert_ = bertscore_metric.compute(predictions=pred, references=refer, lang="en")
        
    #     if (inference):
    #         print(f'BERT score precision is {np.mean(np.array(bert["precision"]))}, recall is {np.mean(np.array(bert["recall"]))}, and F1 is {np.mean(np.array(bert["f1"]))}.')
    #     else:
    #         print(f'{"-"*20} Ground truth upper bound {"-"*20}')
    #         print(len(bert["precision"]))
    #         print(f'BERT score precision is {np.mean(np.array(bert["precision"]))}, recall is {np.mean(np.array(bert["recall"]))}, and F1 is {np.mean(np.array(bert["f1"]))}.')
    #         print(f'{"-"*20} Random selection lower bound {"-"*20}')
    #         print(f'BERT score precision is {np.mean(np.array(bert_["precision"]))}, recall is {np.mean(np.array(bert_["recall"]))}, and F1 is {np.mean(np.array(bert_["f1"]))}.')