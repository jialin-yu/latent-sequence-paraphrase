from distutils.command.config import config
import time
from data_utils import *
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import torch.nn.functional as F
import torch
import random
import numpy as np
import math
import os
from transformer import Transformer

class Trainer(object):

    def __init__(self, configs):
        
        self.configs = configs
        self.device = 'cuda' if self.configs.cuda == True else 'cpu' 
        
        if self.configs.data == 'quora':
            train_data, valid_data, test_data, vocab = self._build_quora_data(self.configs.max_vocab, 
                                        self.configs.train_size, self.configs.valid_size, self.configs.test_size)
            self.configs.max_len = self.configs.quora_max_len
        elif self.configs.data == 'mscoco':
            train_data, valid_data, test_data, vocab = self._build_mscoco_data(self.configs.max_vocab, 
                                        self.configs.train_size, self.configs.valid_size, self.configs.test_size)
            self.configs.max_len = self.configs.quora_max_len
        else:
            print(f'Dataset: {configs.data} not defined.')
            return
        
        self.train_data = train_data
        self.valid_data = valid_data
        self.test_data = test_data
        self.vocab = vocab
        self.configs.vocab_size = len(self.vocab)

        self.model = Transformer(self.configs)
        self.model.to(self.device)
        
    
    def main_lm(self):
        model_dir = self.configs.lm_dir
        max_epoch = self.configs.lm_max_epoch
        lr = self.configs.lm_lr
        grad_clip = self.configs.gradient_clip
        seed = self.configs.seed
        hard = self.configs.lm_hard

        self._set_experiment_seed(seed)

        self.model.prior.apply(self._xavier_initialize)
        optimizer = torch.optim.Adam(self.model.prior.parameters(), lr = lr)
        
        EXP_START_TIME = time.time()
        best_valid_loss = float('inf')
        for epoch in range(max_epoch):
            start_time = time.time()
            train_loss = self._train_lm(self.model.prior, self.train_data, optimizer, grad_clip)
            valid_loss = self._evaluate_lm(self.model.prior, self.valid_data)
            elapsed = time.time() - start_time 
            print(f'Epoch {epoch + 1}/{max_epoch} training done in {elapsed}s.')
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss 
                self._save_model(self.model.prior, model_dir, seed)
                print(f'Save model in epoch {epoch + 1}.')
        
        print('Retrieve best model for testing')
        self._load_model(self.model.prior, model_dir, seed)
        valid_loss = self._evaluate_lm(self.model.prior, self.test_data)
        print(f'Language model done in {time.time() - EXP_START_TIME}s.')

    def train(self, **kwags):
        pass

    def train_mix(self, **kwags):
        max_epoch = max(self.configs.lm_max_epoch, self.configs.vae_max_epoch)
        pass

    def evaluate(self, **kwags):
        pass

    def test(self, **kwags):
        pass

    def _batchify(self, batch):
        src_list, trg_list, src_lens = [], [], []
        for (src_idx, trg_idx) in batch:
            src = torch.tensor([self.configs.bos_id] + src_idx + [self.configs.eos_id], dtype=torch.int64)
            trg = torch.tensor([self.configs.bos_id] + trg_idx + [self.configs.eos_id], dtype=torch.int64)
            src_list.append(src)
            trg_list.append(trg)
        src_ = pad_sequence(src_list, batch_first=True, padding_value=self.configs.pad_id)
        trg_ = pad_sequence(trg_list, batch_first=True, padding_value=self.configs.pad_id)
        return src_.to(self.device), trg_.to(self.device)

    def _build_quora_data(self, max_vocab, train_size, valid_size, test_size):
        sentence_pairs = process_quora(self.configs.quora_fp)
        train_valid_split, test_split = sentence_pairs[:-test_size], sentence_pairs[-test_size:]
        print(f'Calculate origional stats.')
        calculate_stats(train_valid_split)
        VOCAB_ = create_vocab(train_valid_split, self.configs.quora_min_freq, max_vocab)
        VOCAB = append_special_tokens(VOCAB_, self.configs.special_token, self.configs.unk_id)
        train_valid_idx, test_idx = normalise(train_valid_split, test_split, VOCAB, self.configs.quora_max_len)
        print(f'Calculate normalised stats.')
        calculate_stats(train_valid_idx)
        print('Calculate bound for test data')
        calculate_bound(test_split, True, True)
        train_idx = train_valid_idx[:train_size]
        valid_idx = train_valid_idx[-valid_size:]
        train_dl = DataLoader(train_idx, batch_size=self.configs.batch_size, shuffle=True, collate_fn=self._batchify)
        valid_dl = DataLoader(valid_idx, batch_size=self.configs.batch_size, shuffle=False, collate_fn=self._batchify)
        test_dl = DataLoader(test_idx, batch_size=self.configs.batch_size, shuffle=False, collate_fn=self._batchify)

        return train_dl, valid_dl, test_dl, VOCAB

    def _build_mscoco_data(self, source_file_path_train, source_file_path_test, max_len, min_freq, max_vocab):
        sentence_pairs_train_valid = process_mscoco(source_file_path_train)
        sentence_pairs_test = process_mscoco(source_file_path_test)
    

    def _train_lm(self, lm_model, dataloader, optimizer, grad_clip):
        lm_model.train()
        epoch_loss = 0
        start_time = time.time()
        # dynamic log interval
        log_inter = len(dataloader) // 10
        for idx, (src, trg) in enumerate(tqdm(dataloader)):
            optimizer.zero_grad()
            src_m, trg_m, memory_m, src_kp_m, trg_kp_m = self.model._get_mask(src[:, :-1], trg)
            output = lm_model.hard_lm(src[:, :-1], src_m, src_kp_m)
            if self.configs.lm_hard:
                loss = self.model._reconstruct_error(output, src[:, :-1], True, src_kp_m)
            else:
                loss = self.model._reconstruct_error(output, src[:, :-1])
            
            batch_loss = torch.mean(loss)
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(lm_model.parameters(), grad_clip)
            optimizer.step()
            epoch_loss += batch_loss.item()
            if idx % log_inter == 0 and idx > 0:
                print(f'| Batches: {idx}/{len(dataloader)} | Running Loss: {epoch_loss/(idx+1)} | PPL: {math.exp(epoch_loss/(idx+1))} |')
        elapsed = time.time() - start_time
        print(f'Epoch training time is: {elapsed}s.')
        return epoch_loss / len(dataloader)
    
    def _evaluate_lm(self, lm_model, dataloader):
        lm_model.eval()
        epoch_loss = 0
        start_time = time.time()
        for idx, (src, trg) in enumerate(tqdm(dataloader)):
            src_m, trg_m, memory_m, src_kp_m, trg_kp_m = self.model._get_mask(src[:, :-1], trg)
            output = lm_model.hard_lm(src[:, :-1], src_m, src_kp_m)
            if self.configs.lm_hard:
                loss = self.model._reconstruct_error(output, src[:, :-1], True, src_kp_m)
            else:
                loss = self.model._reconstruct_error(output, src[:, :-1])
            
            batch_loss = torch.mean(loss)
            epoch_loss += batch_loss.item()
            
        print(f'| Total Loss: {epoch_loss/(idx+1)} | PPL: {math.exp(epoch_loss/(idx+1))} |')
        elapsed = time.time() - start_time
        print(f'Epoch Evaluation time is: {elapsed}s.')
        return epoch_loss / len(dataloader)
    
    def _xavier_initialize(self, model):
        if hasattr(model, 'weight') and model.weight.dim() > 1:
            nn.init.xavier_uniform_(model.weight.data)
    
    def _count_parameters(self, model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    def _save_model(self, model, model_dir, seed):
        torch.save(model.state_dict(), f'{model_dir}/SEED_{seed}.pt')
    
    def _load_model(self, model, model_dir, seed):
        model.load_state_dict(torch.load(f'{model_dir}/SEED_{seed}.pt'))

    def _set_experiment_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        os.environ['PYTHONHASHSEED'] = str(seed)