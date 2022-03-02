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
from sample import straight_through_softmax

class Trainer(object):

    def __init__(self, configs):
        
        self.configs = configs
        self.device = 'cuda' if self.configs.cuda == True else 'cpu' 
        
        if self.configs.data == 'quora':
            train_data, valid_data, test_data, vocab = self._build_quora_data(self.configs.max_vocab, 
                                        self.configs.train_size, self.configs.valid_size, self.configs.test_size)
        elif self.configs.data == 'mscoco':
            train_data, valid_data, test_data, vocab = self._build_mscoco_data(self.configs.max_vocab, 
                                        self.configs.train_size, self.configs.valid_size, self.configs.test_size)
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
        grad_clip = self.configs.lm_gc
        seed = self.configs.seed
        lm_name = self.configs.lm_id

        print(f'Set LM experiment seed as: {seed}')
        self._set_experiment_seed(seed)

        self.model.prior.apply(self._xavier_initialize)
        optimizer = torch.optim.Adam(self.model.prior.parameters(), lr = lr)
        
        EXP_START = time.time()
        best_valid_loss = float('inf')
        for epoch in range(max_epoch):
            train_loss = self._train_lm(self.train_data, optimizer, grad_clip)
            valid_loss = self._evaluate_lm(self.valid_data)
            print(f'{"-"*20} Epoch {epoch + 1}/{max_epoch} training done {"-"*20}')
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss 
                self._save_model(self.model, model_dir, lm_name)
                print(f'Save model in epoch {epoch + 1}.')
        
        print('Retrieve best model for testing')
        self._load_model(self.model, model_dir, seed, lm_name)
        self.model.to(self.device)
        valid_loss = self._evaluate_lm(self.test_data)
        print(f'LM experiment done in {time.time() - EXP_START}s.')

    def main_vae(self):
        lm_dir = self.configs.lm_dir
        model_dir = self.configs.vae_dir
        max_epoch = self.configs.lm_max_epoch
        lr = self.configs.lm_lr
        grad_clip = self.configs.lm_gc
        seed = self.configs.seed

        print(f'Set LM experiment seed as: {seed}')
        self._set_experiment_seed(seed)

        self.model.prior.apply(self._xavier_initialize)
        optimizer = torch.optim.Adam(self.model.prior.parameters(), lr = lr)
        
        EXP_START = time.time()
        best_valid_loss = float('inf')
        for epoch in range(max_epoch):
            train_loss = self._train_lm(self.model.prior, self.train_data, optimizer, grad_clip)
            valid_loss = self._evaluate_lm(self.model.prior, self.valid_data)
            print(f'{"-"*20} Epoch {epoch + 1}/{max_epoch} training done {"-"*20}')
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss 
                self._save_model(self.model.prior, model_dir, seed)
                print(f'Save model in epoch {epoch + 1}.')
        
        print('Retrieve best model for testing')
        self._load_model(self.model.prior, model_dir, seed)
        self.model.prior.to(self.device)
        valid_loss = self._evaluate_lm(self.model.prior, self.test_data)
        print(f'LM experiment done in {time.time() - EXP_START}s.')

    def train_mix(self, **kwags):
        max_epoch = max(self.configs.lm_max_epoch, self.configs.vae_max_epoch)
        pass

    def evaluate(self, **kwags):
        pass

    def test(self, **kwags):
        pass

    def _batchify(self, batch):
        src_list, trg_list, src_list_ = [], [], []
        for (src_idx, trg_idx, src_idx_) in batch:
            src = torch.tensor([self.configs.bos_id] + src_idx + [self.configs.eos_id], dtype=torch.int64)
            trg = torch.tensor([self.configs.bos_id] + trg_idx + [self.configs.eos_id], dtype=torch.int64)
            src_ = torch.tensor([self.configs.bos_id] + src_idx_ + [self.configs.eos_id], dtype=torch.int64)
            src_list.append(src)
            trg_list.append(trg)
            src_list_.append(src_)
        src_ = pad_sequence(src_list, batch_first=True, padding_value=self.configs.pad_id)
        trg_ = pad_sequence(trg_list, batch_first=True, padding_value=self.configs.pad_id)
        src__ = pad_sequence(src_list_, batch_first=True, padding_value=self.configs.pad_id)
        return src_.to(self.device), trg_.to(self.device), src__.to(self.device)

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
    

    def _train_lm(self, dataloader, optimizer, grad_clip):
        self.model.prior.train()
        epoch_loss = 0
        start_time = time.time()
        
        log_inter = len(dataloader) // 5
        for idx, (src, trg, src_) in enumerate(tqdm(dataloader)):
            optimizer.zero_grad()

            if self.configs.unsupervised:
                
                if self.configs.use_pseudo:
                    src_m, trg_m, memory_m, src_kp_m, trg_kp_m = self.model._get_mask(src_[:, :-1], trg)        
                    output = self.model.prior.hard_lm(src_[:, :-1], src_m, src_kp_m)           
                    b_loss = self.model._batch_reconstruct_error(output, src_[:, 1:], self.configs.hard, src_kp_m)
                    i_loss = self.model._instance_reconstruct_error(output, src_[:, 1:], self.configs.hard, src_kp_m)
                else:
                    src_m, trg_m, memory_m, src_kp_m, trg_kp_m = self.model._get_mask(src[:, :-1], trg)        
                    output = self.model.prior.hard_lm(src[:, :-1], src_m, src_kp_m)           
                    b_loss = self.model._batch_reconstruct_error(output, src[:, 1:], self.configs.hard, src_kp_m)
                    i_loss = self.model._instance_reconstruct_error(output, src[:, 1:], self.configs.hard, src_kp_m)
            else:
                src_m, trg_m, memory_m, src_kp_m, trg_kp_m = self.model._get_mask(trg[:, :-1], trg)        
                output = self.model.prior.hard_lm(trg[:, :-1], src_m, src_kp_m)
                b_loss = self.model._batch_reconstruct_error(output, trg[:, 1:], self.configs.hard, src_kp_m)
                i_loss = self.model._instance_reconstruct_error(output, trg[:, 1:], self.configs.hard, src_kp_m)
            
            b_loss = torch.mean(b_loss)
            i_loss = torch.mean(i_loss)

            if self.configs.batch_loss: 
                b_loss.backward()
            else:
                i_loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.prior.parameters(), grad_clip)
            optimizer.step()

            epoch_loss += i_loss.item()
            if idx % log_inter == 0 and idx > 0:
                print(f'| Batches: {idx}/{len(dataloader)} | Running Loss: {epoch_loss/(idx+1)} | PPL: {math.exp(epoch_loss/(idx+1))} |')
        elapsed = time.time() - start_time
        print(f'Epoch training time is: {elapsed}s.')
        return epoch_loss / len(dataloader)
    
    def _evaluate_lm(self, dataloader):
        self.model.prior.eval()
        epoch_loss = 0
        start_time = time.time()
        for idx, (src, trg, src_) in enumerate(tqdm(dataloader)):
            
            if self.configs.unsupervised:
                if self.configs.use_pseudo:
                    src_m, trg_m, memory_m, src_kp_m, trg_kp_m = self.model._get_mask(src_[:, :-1], trg)
                    output = self.model.prior.hard_lm(src_[:, :-1], src_m, src_kp_m)
                    loss = self.model._instance_reconstruct_error(output, src_[:, 1:], self.configs.hard, src_kp_m)
                else:
                    src_m, trg_m, memory_m, src_kp_m, trg_kp_m = self.model._get_mask(src[:, :-1], trg)
                    output = self.model.prior.hard_lm(src[:, :-1], src_m, src_kp_m)
                    loss = self.model._instance_reconstruct_error(output, src[:, 1:], self.configs.hard, src_kp_m)
            else:
                src_m, trg_m, memory_m, src_kp_m, trg_kp_m = self.model._get_mask(trg[:, :-1], trg)
                output = self.model.prior.hard_lm(trg[:, :-1], src_m, src_kp_m)
                loss = self.model._instance_reconstruct_error(output, trg[:, 1:], self.configs.hard, src_kp_m)
                
            batch_loss = torch.mean(loss)
            epoch_loss += batch_loss.item()
            
        print(f'| Total Loss: {epoch_loss/(idx+1)} | PPL: {math.exp(epoch_loss/(idx+1))} |')
        elapsed = time.time() - start_time
        print(f'Epoch Evaluation time is: {elapsed}s.')
        return epoch_loss / len(dataloader)

    def _train_vae(self, dataloader, optimizer, grad_clip, temperature=None):
        self.model.train()
        self.model.prior.eval() # freeze parameter for prior
        epoch_total_loss, epoch_rec_loss, epoch_kl_loss = 0
        start_time = time.time()
        
        log_inter = len(dataloader) // 5
        for idx, (src, trg, src_) in enumerate(tqdm(dataloader)):
            optimizer.zero_grad()

            if self.configs.unsupervised: 

                if self.configs.use_pseudo:
                    # trg_ (B, S, V)
                    trg_ = self.model.encode_sample_decode(self.model.src_encoder, self.model.trg_decoder, 
                        src, src_, self.configs.hard, self.configs.gumbel_max, temperature) 
                else:
                    trg_ = self.model.encode_sample_decode(self.model.src_encoder, self.model.trg_decoder, 
                        src, src, self.configs.hard, self.configs.gumbel_max, temperature)
                
                if self.configs.use_pretrain_lm:
                    
                    if self.configs.use_pseudo:
                        src_m, trg_m, memory_m, src_kp_m, trg_kp_m = self.model._get_mask(src_[:, :-1], trg) 
                        p_trg = self.model.prior.hard_lm(src_[:, :-1], src_m, src_kp_m)
                    else:
                        src_m, trg_m, memory_m, src_kp_m, trg_kp_m = self.model._get_mask(src[:, :-1], trg) 
                        p_trg = self.model.prior.hard_lm(src[:, :-1], src_m, src_kp_m)
                else:
                    if self.configs.use_pseudo:
                        src_m, trg_m, memory_m, src_kp_m, trg_kp_m = self.model._get_mask(src_[:, :-1], trg) 
                        p_trg = F.one_hot(src_[:, 1:], self.configs.vocab_size)
                    else:
                        src_m, trg_m, memory_m, src_kp_m, trg_kp_m = self.model._get_mask(src[:, :-1], trg) 
                        p_trg = F.one_hot(src[:, 1:], self.configs.vocab_size)

            else:
                trg_ = self.model.encode_sample_decode(self.model.src_encoder, self.model.trg_decoder, 
                    src, trg, self.configs.hard, self.configs.gumbel_max, temperature)
                
                if self.configs.use_pretrain_lm:
                    
                    src_m, trg_m, memory_m, src_kp_m, trg_kp_m = self.model._get_mask(trg[:, :-1], trg) 
                    p_trg = self.model.prior.hard_lm(trg[:, :-1], src_m, src_kp_m)
                else:
                    src_m, trg_m, memory_m, src_kp_m, trg_kp_m = self.model._get_mask(trg[:, :-1], trg) 
                    p_trg = F.one_hot(trg_[:, 1:], self.configs.vocab_size)
                
            
            q_trg = F.softmax(trg_[:, 1:, :], dim=-1)

             
            b_kl_loss = self.model._batch_cross_entropy(q_trg, p_trg, src_kp_m, self.configs.hard)
            i_kl_loss = self.model._instance_cross_entropy(q_trg, p_trg, src_kp_m, self.configs.hard)

            # reconstruct the origional sequence
                
            src__ = self.model.encode_and_decode(self.model.trg_encoder, self.model.src_decoder,
                trg_, src, encode_hard=False)
                
            src_m, trg_m, memory_m, src_kp_m, trg_kp_m = self.model._get_mask(src[:, :-1], trg)     ?   
                         
            b_rec_loss = self.model._batch_reconstruct_error(src__, src[:, 1:], self.configs.hard, src_kp_m)
            i_rec_loss = self.model._instance_reconstruct_error(src__, src[:, 1:], self.configs.hard, src_kp_m)

            if self.configs.use_pretrain_lm:

                p_trg = pass
            
           
                src_m, trg_m, memory_m, src_kp_m, trg_kp_m = self.model._get_mask(trg[:, :-1], trg)        
                output = model.hard_lm(trg[:, :-1], src_m, src_kp_m)

                if self.configs.hard:
                    b_loss = self.model._batch_reconstruct_error(output, trg[:, 1:], True, src_kp_m)
                    i_loss = self.model._instance_reconstruct_error(output, trg[:, 1:], True, src_kp_m)
                else:
                    b_loss = self.model._batch_reconstruct_error(output, trg[:, 1:])
                    i_loss = self.model._instance_reconstruct_error(output, trg[:, 1:])  
            
            b_loss = torch.mean(b_loss)
            i_loss = torch.mean(i_loss)

            if self.configs.batch_loss: 
                b_loss.backward()
            else:
                i_loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            epoch_loss += i_loss.item()
            if idx % log_inter == 0 and idx > 0:
                print(f'| Batches: {idx}/{len(dataloader)} | Running Loss: {epoch_loss/(idx+1)} | PPL: {math.exp(epoch_loss/(idx+1))} |')
        elapsed = time.time() - start_time
        print(f'Epoch training time is: {elapsed}s.')
        return epoch_loss / len(dataloader)

    
    def _xavier_initialize(self, model):
        if hasattr(model, 'weight') and model.weight.dim() > 1:
            nn.init.xavier_uniform_(model.weight.data)
    
    def _count_parameters(self, model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    def _save_model(self, model, model_dir, seed, name):
        torch.save(model.state_dict(), f'{model_dir}/{name}.pt')
    
    def _load_model(self, model, model_dir, seed, name):
        model.load_state_dict(torch.load(f'{model_dir}/{name}.pt'))

    def _set_experiment_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        os.environ['PYTHONHASHSEED'] = str(seed)