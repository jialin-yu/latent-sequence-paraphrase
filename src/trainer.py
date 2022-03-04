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

from pipeline import index_to_token, remove_bos_eos

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
        print(f'{"-"*20} Set model as {self._count_parameters(self.model)} parameters {"-"*20}')
    
    def main_inference(self, dataloader, vocab):

        src_trg_idx = []

        for idx, (src, trg, src_) in enumerate(tqdm(dataloader)):

            trg_batch = self.model.inference(self.model.src_encoder, self.model.trg_decoder, src, 30)

            for index, _ in enumerate(trg_batch):
                src_trg_idx.append((trg_batch[index].cpu(), trg[index].cpu()))
        
        test_ = [(index_to_token(remove_bos_eos(i, self.configs.bos_id, self.configs.eos_id), vocab), index_to_token(remove_bos_eos(j, self.configs.bos_id, self.configs.eos_id), vocab)) for (i, j) in src_trg_idx]
        print(f'{"-"*20} Calculate Final Results {"-"*20}')
        calculate_bound(test_, True)



    
    def main_lm(self):
        model_dir = self.configs.lm_dir
        max_epoch = self.configs.lm_max_epoch
        lr = self.configs.lm_lr
        grad_clip = self.configs.gc
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
        self._load_model(self.model, model_dir, lm_name)
        self.model.to(self.device)
        valid_loss = self._evaluate_lm(self.test_data)
        print(f'LM experiment done in {time.time() - EXP_START}s.')
    
    def main_seq2seq(self):
        model_dir = self.configs.seq2seq_dir
        max_epoch = self.configs.seq2seq_max_epoch
        lr = self.configs.seq2seq_lr
        grad_clip = self.configs.gc
        seed = self.configs.seed
        seq2seq_name = self.configs.seq2seq_id

        print(f'Set seq2seq experiment seed as: {seed}')
        self._set_experiment_seed(seed)

        self.model.prior.apply(self._xavier_initialize)
        optimizer = torch.optim.Adam(self.model.prior.parameters(), lr = lr)
        
        EXP_START = time.time()
        best_valid_loss = float('inf')
        for epoch in range(max_epoch):
            train_loss = self._train_seq2seq(self.train_data, optimizer, grad_clip)
            valid_loss = self._evaluate_seq2seq(self.valid_data)
            print(f'{"-"*20} Epoch {epoch + 1}/{max_epoch} training done {"-"*20}')
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss 
                self._save_model(self.model, model_dir, seq2seq_name)
                print(f'Save model in epoch {epoch + 1}.')
        
        print('Retrieve best model for testing')
        self._load_model(self.model, model_dir, seq2seq_name)
        self.model.to(self.device)
        valid_loss = self._evaluate_seq2seq(self.test_data)
        print(f'Seq2seq experiment done in {time.time() - EXP_START}s.')

        self.main_inference(self.test_data, self.vocab)

    def main_vae(self):
        lm_dir = self.configs.lm_dir
        lm_name = self.configs.lm_id
        
        if self.configs.use_pretrain_lm:
            self._load_model(self.model, lm_dir, lm_name)

        model_dir = self.configs.vae_dir
        vae_name = self.configs.vae_id
        max_epoch = self.configs.vae_max_epoch
        lr = self.configs.vae_lr
        grad_clip = self.configs.gc
        seed = self.configs.seed
        
        factor = 1
        high_temp = 3
        low_temp = 0.5
        temperature = list(np.linspace(high_temp, low_temp, max_epoch))

        print(f'Set VAE experiment seed as: {seed}')
        self._set_experiment_seed(seed)

        self.model.prior.apply(self._xavier_initialize)
        optimizer = torch.optim.Adam(self.model.parameters(), lr = lr)
        
        EXP_START = time.time()
        best_valid_loss = float('inf')
        for epoch in range(max_epoch):
            train_loss = self._train_vae(self.train_data, optimizer, grad_clip, temperature[epoch], factor)
            
            print(f'{"-"*20} Epoch {epoch + 1}/{max_epoch} training done {"-"*20}')

            valid_loss = self._evaluate_vae(self.valid_data, low_temp)
            
            print(f'{"-"*20} Epoch {epoch + 1}/{max_epoch} evaluation done {"-"*20}')
            
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss 
                self._save_model(self.model, model_dir, vae_name)
                print(f'Save model in epoch {epoch + 1}.')
        
        print(f'{"-"*40}')
        print('Retrieve best model for testing')
        self._load_model(self.model, model_dir, vae_name)
        self.model.prior.to(self.device)
        valid_loss = self._evaluate_vae(self.test_data, low_temp)
        print(f'VAE experiment done in {time.time() - EXP_START}s.')

    def main_semi_supervised(self):
        lm_dir = self.configs.lm_dir
        lm_name = self.configs.lm_id
        
        if self.configs.use_pretrain_lm:
            self._load_model(self.model, lm_dir, lm_name)

        model_dir = self.configs.semi_dir
        vae_name = self.configs.vae_id
        max_epoch = self.configs.vae_max_epoch
        lr = self.configs.vae_lr
        grad_clip = self.configs.gc
        seed = self.configs.seed

        factor = 1
        high_temp = 3
        low_temp = 0.5
        temperature = list(np.linspace(high_temp, low_temp, max_epoch))

        print(f'Set SEMI experiment seed as: {seed}')
        self._set_experiment_seed(seed)

        self.model.prior.apply(self._xavier_initialize)
        optimizer = torch.optim.Adam(self.model.parameters(), lr = lr)
        
        EXP_START = time.time()
        best_valid_loss = float('inf')
        for epoch in range(max_epoch):
            train_loss = self._train_vae(self.train_data, optimizer, grad_clip, temperature[epoch], factor)
            train_loss += self._train_seq2seq(self.train_data, optimizer, grad_clip)

            valid_loss = self._evaluate_vae(self.valid_data, low_temp)
            valid_loss += self._evaluate_seq2seq(self.valid_data)
            
            print(f'{"-"*20} Epoch {epoch + 1}/{max_epoch} training done {"-"*20}')
            
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss 
                self._save_model(self.model, model_dir, vae_name)
                print(f'Save model in epoch {epoch + 1}.')
        
        print('Retrieve best model for testing')
        self._load_model(self.model, model_dir, vae_name)
        self.model.prior.to(self.device)
        valid_loss = self._evaluate_vae(self.test_data, low_temp)
        valid_loss += self._evaluate_seq2seq(self.test_data)
        print(f'VAE experiment done in {time.time() - EXP_START}s.')

    def main_enhance(self):
        
        print(f'Set enhance experiment')

        vae_name = self.configs.vae_id
        vae_model_dir = self.configs.vae_dir
        self.main_vae()
        self._load_model(self.model, vae_model_dir, vae_name)

        model_dir = self.configs.enhance_dir
        max_epoch = self.configs.seq2seq_max_epoch
        lr = self.configs.seq2seq_lr
        grad_clip = self.configs.gc
        seed = self.configs.seed
        seq2seq_name = self.configs.seq2seq_id

        optimizer = torch.optim.Adam(self.model.prior.parameters(), lr = lr)
        
        best_valid_loss = float('inf')
        for epoch in range(max_epoch):
            train_loss = self._train_seq2seq(self.train_data, optimizer, grad_clip)
            valid_loss = self._evaluate_seq2seq(self.valid_data)
            print(f'{"-"*20} Epoch {epoch + 1}/{max_epoch} training done {"-"*20}')
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss 
                self._save_model(self.model, model_dir, vae_name)
                print(f'Save model in epoch {epoch + 1}.')
        
        print('Retrieve best model for testing')
        self._load_model(self.model, model_dir, vae_name)
        self.model.to(self.device)
        valid_loss = self._evaluate_seq2seq(self.test_data)
        print(f'Enhance experiment done .')

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
        calculate_bound(test_split, True)
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

                    output = self.model.decode_lm(self.model.prior, src_[:, :-1], True)         
                    b_loss = self.model._batch_reconstruct_error(output, src_[:, 1:], self.configs.hard_loss)
                    i_loss = self.model._instance_reconstruct_error(output, src_[:, 1:], self.configs.hard_loss)

                else:
                    
                    output = self.model.decode_lm(self.model.prior, src[:, :-1], True)         
                    b_loss = self.model._batch_reconstruct_error(output, src[:, 1:], self.configs.hard_loss)
                    i_loss = self.model._instance_reconstruct_error(output, src[:, 1:], self.configs.hard_loss)

            else:
                
                output = self.model.decode_lm(self.model.prior, trg[:, :-1], True)         
                b_loss = self.model._batch_reconstruct_error(output, trg[:, 1:], self.configs.hard_loss)
                i_loss = self.model._instance_reconstruct_error(output, trg[:, 1:], self.configs.hard_loss)
            
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

                    output = self.model.decode_lm(self.model.prior, src_[:, :-1], True)         
                    loss = self.model._instance_reconstruct_error(output, src_[:, 1:], self.configs.hard_loss)

                else:

                    output = self.model.decode_lm(self.model.prior, src[:, :-1], True)         
                    loss = self.model._instance_reconstruct_error(output, src[:, 1:], self.configs.hard_loss)
            else:

                output = self.model.decode_lm(self.model.prior, trg[:, :-1], True)         
                loss = self.model._instance_reconstruct_error(output, trg[:, 1:], self.configs.hard_loss)
                
            batch_loss = torch.mean(loss)
            epoch_loss += batch_loss.item()
            
        print(f'| Total Loss: {epoch_loss/(idx+1)} | PPL: {math.exp(epoch_loss/(idx+1))} |')
        elapsed = time.time() - start_time
        print(f'Epoch Evaluation time is: {elapsed}s.')
        return epoch_loss / len(dataloader)

    def _train_vae(self, dataloader, optimizer, grad_clip, temperature=None, factor=1):
        self.model.train()
        self.model.prior.eval() # freeze parameter for prior
        epoch_total_loss, epoch_rec_loss, epoch_kl_loss = 0, 0, 0
        start_time = time.time()
        
        log_inter = len(dataloader) // 5
        for idx, (src, trg, src_) in enumerate(tqdm(dataloader)):
            optimizer.zero_grad()

            '''
            reconstruction of src

            sampling based decode, return trg_ (B, S, V) in normalised form
            
            '''
            if self.configs.unsupervised: 

                if self.configs.use_pseudo:
                    # trg_ (B, S, V)
                    trg_ = self.model.encode_sample_decode(self.model.src_encoder, self.model.trg_decoder, 
                        src, src_, self.configs.latent_hard, self.configs.gumbel_max, temperature) 
                else:
                    trg_ = self.model.encode_sample_decode(self.model.src_encoder, self.model.trg_decoder, 
                        src, src, self.configs.latent_hard, self.configs.gumbel_max, temperature)
            else:

                trg_ = self.model.encode_sample_decode(self.model.src_encoder, self.model.trg_decoder, 
                    src, trg, self.configs.latent_hard, self.configs.gumbel_max, temperature)
            
            # src__ (B, S-1, V)
            src__ = self.model.encode_and_decode(self.model.trg_encoder, self.model.src_decoder,
                trg_, src[:,1:], False)
            
            # calculate reconstruction loss between src and src__
            if self.configs.unsupervised: 
                
                if self.configs.use_pretrain_lm:
                
                    if self.configs.use_pseudo:

                        p_trg = self.model.decode_lm(self.model.prior, src_[:, :-1], True)
                        p_trg_ = src_[:, 1:]
                    
                    else:
                    
                        p_trg = self.model.decode_lm(self.model.prior, src[:, :-1], True)
                        p_trg_ = src[:, 1:]

                else:

                    if self.configs.use_pseudo:

                        p_trg = F.one_hot(src_[:, 1:], self.configs.vocab_size)
                        p_trg_ = src_[:, 1:]
                    
                    else:
                    
                        p_trg = F.one_hot(src[:, 1:], self.configs.vocab_size)
                        p_trg_ = src[:, 1:]

            else:

                if self.configs.use_pretrain_lm:
                    
                    p_trg = self.model.decode_lm(self.model.prior, trg[:, :-1], True)
                
                else:
                    
                    p_trg = F.one_hot(trg[:, 1:], self.configs.vocab_size)
                
                p_trg_ = trg[:, 1:]

            q_trg = trg_[:, 1:]


            b_rec_loss = self.model._batch_reconstruct_error(src__, src[:, 1:], self.configs.hard_loss)
            i_rec_loss = self.model._instance_reconstruct_error(src__, src[:, 1:], self.configs.hard_loss)
             
            b_kl_loss = self.model._batch_cross_entropy(q_trg, p_trg, p_trg_, self.configs.hard_loss)
            i_kl_loss = self.model._instance_cross_entropy(q_trg, p_trg, p_trg_, self.configs.hard_loss)

            # print(i_kl_loss.size())

            # print(i_rec_loss.size())

            b_loss = torch.mean(b_rec_loss) + torch.mean(factor * b_kl_loss)
            i_loss = torch.mean(i_rec_loss) + torch.mean(factor * i_kl_loss)

            if self.configs.batch_loss: 
                b_loss.backward()
            else:
                i_loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
            optimizer.step()

            epoch_total_loss += i_loss.item()
            epoch_rec_loss += torch.mean(i_rec_loss).item()
            epoch_kl_loss += torch.mean(i_kl_loss).item()

            if idx % log_inter == 0 and idx > 0:
                print(f'| Batches: {idx}/{len(dataloader)} | PPL: {math.exp(epoch_rec_loss/(idx+1))} |')
                print(f'| REC Loss: {epoch_rec_loss/(idx+1)} | KL Loss: {epoch_kl_loss/(idx+1)} | Total Loss: {epoch_total_loss/(idx+1)} |')
        
        elapsed = time.time() - start_time
        print(f'Epoch training time is: {elapsed}s.')
        return epoch_total_loss / len(dataloader)

    def _evaluate_vae(self, dataloader, temperature=None, factor=1):
        self.model.eval()
        epoch_total_loss, epoch_rec_loss, epoch_kl_loss = 0, 0, 0
        start_time = time.time()
        
        log_inter = len(dataloader) // 5
        for idx, (src, trg, src_) in enumerate(tqdm(dataloader)):

            '''
            reconstruction of src

            sampling based decode, return trg_ (B, S, V) in normalised form
            
            '''
            if self.configs.unsupervised: 

                if self.configs.use_pseudo:
                    # trg_ (B, S, V)
                    trg_ = self.model.encode_sample_decode(self.model.src_encoder, self.model.trg_decoder, 
                        src, src_, self.configs.latent_hard, self.configs.gumbel_max, temperature) 
                else:
                    trg_ = self.model.encode_sample_decode(self.model.src_encoder, self.model.trg_decoder, 
                        src, src, self.configs.latent_hard, self.configs.gumbel_max, temperature)
            else:

                trg_ = self.model.encode_sample_decode(self.model.src_encoder, self.model.trg_decoder, 
                    src, trg, self.configs.latent_hard, self.configs.gumbel_max, temperature)
            
            # src__ (B, S-1, V)
            src__ = self.model.encode_and_decode(self.model.trg_encoder, self.model.src_decoder,
                trg_, src[:,1:], False)
            
            # calculate reconstruction loss between src and src__
            if self.configs.unsupervised: 
                
                if self.configs.use_pretrain_lm:
                
                    if self.configs.use_pseudo:

                        p_trg = self.model.decode_lm(self.model.prior, src_[:, :-1], True)
                        p_trg_ = src_[:, 1:]
                    
                    else:
                    
                        p_trg = self.model.decode_lm(self.model.prior, src[:, :-1], True)
                        p_trg_ = src[:, 1:]

                else:

                    if self.configs.use_pseudo:

                        p_trg = F.one_hot(src_[:, 1:], self.configs.vocab_size)
                        p_trg_ = src_[:, 1:]
                    
                    else:
                    
                        p_trg = F.one_hot(src[:, 1:], self.configs.vocab_size)
                        p_trg_ = src[:, 1:]

            else:

                if self.configs.use_pretrain_lm:
                    
                    p_trg = self.model.decode_lm(self.model.prior, trg[:, :-1], True)
                
                else:
                    
                    p_trg = F.one_hot(trg[:, 1:], self.configs.vocab_size)
                
                p_trg_ = trg[:, 1:]

            q_trg = trg_[:, 1:]


            # b_rec_loss = self.model._batch_reconstruct_error(src__, src[:, 1:], self.configs.hard_loss)
            rec_loss = self.model._instance_reconstruct_error(src__, src[:, 1:], self.configs.hard_loss)
             
            # b_kl_loss = self.model._batch_cross_entropy(q_trg, p_trg, p_trg_, self.configs.hard_loss)
            kl_loss = self.model._instance_cross_entropy(q_trg, p_trg, p_trg_, self.configs.hard_loss)
         
            loss = torch.mean(rec_loss) + torch.mean(factor * kl_loss)
            # i_loss = torch.mean(i_rec_loss + factor * i_kl_loss)

            # if self.configs.batch_loss: 
            #     b_loss.backward()
            # else:
            #     i_loss.backward()

            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
            # optimizer.step()

            epoch_total_loss += loss.item()
            epoch_rec_loss += torch.mean(rec_loss).item()
            epoch_kl_loss += torch.mean(kl_loss).item()

        
        print(f'| Batches: {idx}/{len(dataloader)} | PPL: {math.exp(epoch_rec_loss/(idx+1))} |')
        print(f'| REC Loss: {epoch_rec_loss/(idx+1)} | KL Loss: {epoch_kl_loss/(idx+1)} | Total Loss: {epoch_total_loss/(idx+1)} |')
        
        elapsed = time.time() - start_time
        print(f'Epoch evaluation time is: {elapsed}s.')
        return epoch_total_loss / len(dataloader)
    
    def _train_seq2seq(self, dataloader, optimizer, grad_clip):
        self.model.train()
        self.model.prior.eval() # freeze parameter for prior
        epoch_loss = 0
        start_time = time.time()
        
        log_inter = len(dataloader) // 5
        for idx, (src, trg, src_) in enumerate(tqdm(dataloader)):
            optimizer.zero_grad()

            '''
            duo training
            '''
            src_ = self.model.encode_and_decode(self.model.trg_encoder, self.model.src_decoder,
                trg, src[:,1:], True)
            
            trg_ = self.model.encode_and_decode(self.model.src_encoder, self.model.trg_decoder,
                src, trg[:,1:], True)

            b_loss_src = self.model._batch_reconstruct_error(src_, src[:, 1:], self.configs.hard_loss)
            b_loss_trg = self.model._batch_reconstruct_error(trg_, trg[:, 1:], self.configs.hard_loss)

            i_loss_src = self.model._instance_reconstruct_error(src_, src[:, 1:], self.configs.hard_loss)
            i_loss_trg = self.model._instance_reconstruct_error(trg_, trg[:, 1:], self.configs.hard_loss)


            # print(i_kl_loss.size())

            # print(i_rec_loss.size())

            b_loss = torch.mean(b_loss_src) + torch.mean(b_loss_trg)
            i_loss = torch.mean(i_loss_src) + torch.mean(i_loss_trg)

            if self.configs.batch_loss: 
                b_loss.backward()
            else:
                i_loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
            optimizer.step()

            epoch_loss += i_loss.item()

            if idx % log_inter == 0 and idx > 0:
                print(f'| Batches: {idx}/{len(dataloader)} | PPL: {math.exp(epoch_loss/(idx+1)/2)} | LOSS: {epoch_loss/(idx+1)} |')
        
        elapsed = time.time() - start_time
        print(f'Epoch training time is: {elapsed}s.')
        return epoch_loss / len(dataloader)

    def _evaluate_seq2seq(self, dataloader):
        self.model.eval()
        epoch_loss = 0
        start_time = time.time()
        
        log_inter = len(dataloader) // 5
        for idx, (src, trg, src_) in enumerate(tqdm(dataloader)):

            '''
            duo training
            '''
            src_ = self.model.encode_and_decode(self.model.trg_encoder, self.model.src_decoder,
                trg, src[:,1:], True)
            
            trg_ = self.model.encode_and_decode(self.model.src_encoder, self.model.trg_decoder,
                src, trg[:,1:], True)

            b_loss_src = self.model._batch_reconstruct_error(src_, src[:, 1:], self.configs.hard_loss)
            b_loss_trg = self.model._batch_reconstruct_error(trg_, trg[:, 1:], self.configs.hard_loss)

            i_loss_src = self.model._instance_reconstruct_error(src_, src[:, 1:], self.configs.hard_loss)
            i_loss_trg = self.model._instance_reconstruct_error(trg_, trg[:, 1:], self.configs.hard_loss)


            # print(i_kl_loss.size())

            # print(i_rec_loss.size())

            b_loss = torch.mean(b_loss_src) + torch.mean(b_loss_trg)
            i_loss = torch.mean(i_loss_src) + torch.mean(i_loss_trg)

            epoch_loss += i_loss.item()

            
        print(f'| PPL: {math.exp(epoch_loss/(idx+1))} | LOSS: {epoch_loss/(idx+1)} |')
        
        elapsed = time.time() - start_time
        print(f'Epoch evaluateion time is: {elapsed}s.')
        return epoch_loss / len(dataloader)

    def _xavier_initialize(self, model):
        if hasattr(model, 'weight') and model.weight.dim() > 1:
            nn.init.xavier_uniform_(model.weight.data)
    
    def _count_parameters(self, model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    def _save_model(self, model, model_dir, name):
        torch.save(model.state_dict(), f'{model_dir}/{name}.pt')
    
    def _load_model(self, model, model_dir, name):
        model.load_state_dict(torch.load(f'{model_dir}/{name}.pt'))

    def _set_experiment_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        os.environ['PYTHONHASHSEED'] = str(seed)