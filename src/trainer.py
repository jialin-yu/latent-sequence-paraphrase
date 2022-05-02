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
from model import Transformer
from sample import straight_through_softmax

from pipeline import tokenize, decode_index_to_string, stringify, tokenizer
from sklearn.utils import shuffle, resample
import wandb
from os.path import exists


class Trainer(object):

    def __init__(self, configs):
        
        self.configs = configs
        self.configs.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = self.configs.device
        self.tokenizer = tokenizer
        self.bos_id = self.tokenizer.bos_token_id
        self.eos_id = self.tokenizer.eos_token_id
        self.pad_id = self.tokenizer.pad_token_id

        assert self.configs.data in ['quora', 'mscoco']
        
        if self.configs.data == 'quora':
            lm_data, un_data, train_data, valid_data, test_data, test_split = self._build_data(self.configs.un_train_size, 
            self.configs.train_size, self.configs.quora_valid, self.configs.quora_test, data='quora')
        if self.configs.data == 'mscoco':
            lm_data, un_data, train_data, valid_data, test_data, test_split = self._build_data(self.configs.un_train_size,
            self.configs.train_size, self.configs.mscoco_valid, self.configs.mscoco_test, data='mscoco')
        
        self.lm_data = lm_data
        self.un_data = un_data
        self.train_data = train_data
        self.valid_data = valid_data
        self.test_data = test_data
        self.test_split = test_split

        self.configs.vocab_size = self.tokenizer.vocab_size
        self.configs.pad_id = self.pad_id

        print(f'{"-"*20} {self.configs.data} Data Description {"-"*20}') 
        if self.configs.data == 'quora':
            print(f'Use {self.configs.un_train_size} unsupervised data; {self.configs.train_size} training data; {self.configs.quora_valid} validation data and {self.configs.quora_test} testing data.')
        if self.configs.data == 'mscoco':
            print(f'Use {self.configs.un_train_size} unsupervised data; {self.configs.train_size} training data; {self.configs.mscoco_valid} validation data and {self.configs.mscoco_test} testing data.')
        print(f'{"-"*40}')

        self.model = Transformer(self.configs)
        self.model.to(self.device)
        print(f'{"-"*20} Model Description {"-"*20}')
        print(f'Set model as total {self._count_parameters(self.model)} trainable parameters')
        print(f'{"-"*40}')
    
    def main_inference(self, dataloader, test_split, interpolate_latent=True):
        self.model.eval()

        print(f'{"-"*20} Perform inference {"-"*20}')

        pred_idx = []
        first_n = 50
        latent_samples = 5
        for idx, (src, trg) in enumerate(tqdm(dataloader)):      
            trg_ = self.model.inference(src, self.configs.max_len, deterministic=True)
            for trg in trg_:
                pred_idx.append(trg.cpu())

        test_token = [tokenize(decode_index_to_string(pred)) for pred in pred_idx]

        if interpolate_latent:
            print(f'{"-"*20} Perform Latent Sampling {"-"*20}')
            latent_sample_idx = self._latent_interpolate(dataloader, latent_samples)
            latent_token = [[tokenize(decode_index_to_string(s)) for s in sets] for sets in latent_sample_idx]
            # print(test_token[0])
            # print(latent_token[0])
        
        print(f'{"-"*20} Calculate Final Results {"-"*20}')
        calculate_bound(test_token, test_split, True, True, True)
        # if interpolate_latent:
        #     gumble_search_token = gumble_search(test_token, latent_token, test_split)
        #     print(f'{"-"*20} Calculate BERT Gumbel Search Results {"-"*20}')
        #     calculate_bound(gumble_search_token, test_split, True, True, True)
        print(f'{"-"*20} Print first {first_n} Results {"-"*20}')
        test__ = test_token[:first_n]

        if interpolate_latent:
            # latent_sample__ = latent_token[:first_n]
            for index, _ in enumerate(test__):
                print(f'{"-"*20} Print {index + 1} Example {"-"*20}')
                print(f'Source: {stringify(test_split[index][0])}')
                print(f'Prediction: {stringify(test__[index])}')
                ref = test_split[index][1:]
                for reff in ref:
                    print(f'Reference: {stringify(reff)}')
                for latent in latent_token[index]:
                    print(f'Latent Sample: {stringify(latent)}')
                print(f'{"-"*20}')
                # print(f'Best Search Prediction: {stringify(gumble_search_token[index])}')
                # table.add_data(index, stringify(test__[index]), [stringify(reff) for reff in ref], [stringify(latent) for latent in latent_sample_[index]])
            print(f'{"-"*40}')
        else:
            for index, _ in enumerate(test__):
                print(f'{"-"*20} Print {index + 1} Example {"-"*20}')
                print(f'Source: {stringify(test_split[index][0])}')
                print(f'Prediction: {stringify(test__[index])}')
                ref = test_split[index][1:]
                for reff in ref:
                    print(f'Reference: {stringify(reff)}')
                print(f'{"-"*20}')
                # table.add_data(index, stringify(test__[index]), [stringify(reff) for reff in ref])
            print(f'{"-"*40}')
        print(f'{"-"*20} Inference done {"-"*20}')
        
    def _latent_interpolate(self, dataloader, number_samples):
        sample_idx = []
        for _, (src, _) in enumerate(tqdm(dataloader)):
                for index, _ in enumerate(src):
                    temp = []
                    for _ in range(number_samples):
                        trg_ = self.model.inference(src[index].unsqueeze(0), self.configs.max_len, deterministic=False)
                        temp.append(trg_.squeeze().cpu())
                    sample_idx.append(temp)
        return sample_idx

    
    def main_seq2seq(self):
        wandb.init(project='paraphrase-seq2seq', config=self.configs, entity='du_jialin', settings=wandb.Settings(start_method='fork'))
        model_dir = self.configs.seq2seq_dir
        max_epoch = self.configs.seq2seq_max_epoch
        lr = self.configs.seq2seq_lr
        grad_clip = self.configs.grad_clip
        seed = self.configs.seed
        experiment_id = self.configs.seq2seq_id
        standard_seq2seq = self.configs.seq2seq

        print(f'{"-"*40}')
        print(f'{"-"*20} Initialise seq2seq experiment {"-"*20}')
        print(f'Use model directory {model_dir}')
        print(f'Experiment run for max {max_epoch} epochs; total of {len(self.train_data)*max_epoch} steps')
        print(f'Learning rate set as {lr}')
        print(f'Use gradient clip of {grad_clip}')
        print(f'Use seed of {seed}')
        print(f'Experiment id as {experiment_id}')
        print(f'Run standard seq2seq experiment {standard_seq2seq}')
        print(f'Total model parameters {self._count_parameters(self.model.encoder) + self._count_parameters(self.model.decoder)} ')
        print(f'{"-"*40}')

        self._set_experiment_seed(seed)
        self.model.encoder.apply(self._xavier_initialize)
        self.model.decoder.apply(self._xavier_initialize)
        
        params = list(self.model.encoder.parameters()) + list(self.model.decoder.parameters())
        optimizer = torch.optim.Adam(params, lr=lr)

        EXP_START_TIME = time.time()
        best_valid_loss = float('inf')
        for epoch in range(max_epoch):

            if standard_seq2seq:
                train_loss = self._train_seq2seq(self.train_data, optimizer, grad_clip)
            else:
                train_loss = self._train_ddl(self.train_data, optimizer, grad_clip)
            
            valid_loss = self._evaluate_seq2seq(self.valid_data)
            
            wandb.log({'train-loss': train_loss}, step=epoch+1)
            wandb.log({'valid-loss': valid_loss}, step=epoch+1)
            
            print(f'{"-"*20} Epoch {epoch + 1}/{max_epoch} training done {"-"*20}')
            
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss 
                self._save_model(self.model, model_dir, experiment_id)
                print(f'{"-"*20} Save model in epoch {epoch + 1}. {"-"*20}')
        
        print(f'{"-"*40}')
        print(f'{"-"*20} Retrieve best model for testing {"-"*20}')
        self._load_model(self.model, model_dir, experiment_id)
        self.model.to(self.device)
        valid_loss = self._evaluate_seq2seq(self.test_data)
        print(f'Seq2seq experiment done in {time.time() - EXP_START_TIME}s.')
        print(f'{"-"*40}')
        
        self.main_inference(self.test_data, self.test_split, interpolate_latent=False)
    
    

    def main_lm(self):
        wandb.init(project='paraphrase-lm', config=self.configs, entity='du_jialin', settings=wandb.Settings(start_method='fork'))
        model_dir = self.configs.lm_dir
        max_epoch = self.configs.lm_max_epoch
        lr = self.configs.lm_lr
        grad_clip = self.configs.grad_clip
        seed = self.configs.lm_seed
        experiment_id = self.configs.lm_id

        print(f'{"-"*20} Initialise LM experiment {"-"*20}')
        print(f'Use model directory {model_dir}')
        print(f'Experiment run for max {max_epoch} epochs')
        print(f'Learning rate set as {lr}')
        print(f'Use gradient clip of {grad_clip}')
        print(f'Use seed of {seed}')
        print(f'Experiment id as {experiment_id}')
        print(f'Total model parameters {self._count_parameters(self.model.prior)}')
        print(f'{"-"*40}')

        self._set_experiment_seed(seed)
        self.model.prior.apply(self._xavier_initialize)
        params = list(self.model.prior.parameters())
        optimizer = torch.optim.Adam(params, lr = lr)
        
        EXP_START = time.time()
        best_valid_loss = float('inf')
        for epoch in range(max_epoch):
            train_loss = self._train_lm(self.lm_data, optimizer, grad_clip)
            wandb.log({'lm-train-loss': train_loss}, step=epoch+1)
            valid_loss = self._evaluate_lm(self.valid_data)
            wandb.log({'lm-valid-loss': valid_loss}, step=epoch+1)
            print(f'{"-"*20} Epoch {epoch + 1}/{max_epoch} training done {"-"*20}')
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss 
                self._save_model(self.model, model_dir, experiment_id)
                print(f'Save model in epoch {epoch + 1}.')
        
        print('Retrieve best model for testing')
        self._load_model(self.model, model_dir, experiment_id)
        self.model.to(self.device)
        valid_loss = self._evaluate_lm(self.test_data)
        print(f'LM experiment done in {time.time() - EXP_START}s.')

    def main_semi(self):

        wandb.init(project='paraphrase-seq2seq', config=self.configs, entity='du_jialin', settings=wandb.Settings(start_method='fork'))
        lm_dir = self.configs.lm_dir
        lm_id = self.configs.lm_id
        use_lm = self.configs.use_lm
        
        if use_lm:
            lm_name = f'{lm_dir}/{lm_id}.pt'
            assert exists(lm_name)
        
        if use_lm:
            print(f'Use pre-trained LM from dir {lm_dir} and id {lm_id}')
            self._load_model(self.model, lm_dir, lm_id)
        else:
            print(f'No pre-trained LM used')

        model_dir = self.configs.semi_dir
        max_epoch = self.configs.semi_max_epoch
        lr = self.configs.semi_lr
        grad_clip = self.configs.grad_clip
        seed = self.configs.seed
        experiment_id = self.configs.semi_id
        top_k = self.configs.top_k

        # https://iancovert.com/blog/concrete_temperature/
        # if fixed temperature, set to 0.1 as in http://proceedings.mlr.press/v80/chen18j/chen18j.pdf
        # else, set high to 10, low to 0.01 as in http://proceedings.mlr.press/v97/balin19a/balin19a.pdf
        
        if self.configs.fixed_temperature:
            high_temp = 10 # 2K
            low_temp = 10
        else:
            high_temp = 10 # 13K
            low_temp = 1 # 100

            # high_temp = 100
            # low_temp = 10
        
        r_un = self.configs.un_train_size / (self.configs.un_train_size + self.configs.train_size)
        # bigger = self.configs.un_train_size / self.configs.train_size

        beta_factor = 1
        temperature = list(np.linspace(high_temp, low_temp, max_epoch))

        print(f'{"-"*20} Initialise SEMI experiment {"-"*20}')
        print(f'Use model directory {model_dir}')
        print(f'Experiment run for max {max_epoch} epochs')
        print(f'Learning rate set as {lr}')
        print(f'Use gradient clip of {grad_clip}')
        print(f'Use seed of {seed}')
        print(f'Experiment id as {experiment_id}')

        print(f'Total model parameters {self._count_parameters(self.model)} ')
        print(f'Set high temperate as {high_temp} amd low temperature as {low_temp}')
        print(f'Training with beta factor of {beta_factor} for KL')
        print(f'Gumbel Softmax with top {top_k}')
        print(f'{"-"*40}')

        
        self._set_experiment_seed(seed)
        self.model.encoder.apply(self._xavier_initialize)
        self.model.decoder.apply(self._xavier_initialize)

        params = list(self.model.encoder.parameters()) + list(self.model.decoder.parameters())
        # optimizer_un = torch.optim.Adam(params, lr=lr/bigger)
        optimizer = torch.optim.Adam(params, lr=lr)
        
        EXP_START = time.time()
        best_valid_loss = float('inf')
        for epoch in range(max_epoch):
            train_loss_un = self._train_unsupervised(self.un_data, optimizer, grad_clip, temperature[epoch], beta_factor, top_k)
            train_loss_su = self._train_ddl(self.train_data, optimizer, grad_clip)
            # train_loss_su = self._train_unsupervised(self.un_data, optimizer, grad_clip, temperature[epoch], beta_factor, top_k)
            # train_loss = train_loss_un
            
            train_loss = train_loss_un * r_un + train_loss_su * (1 - r_un)
            wandb.log({'train-loss': train_loss}, step=epoch+1)

            valid_loss_ = self._evaluate_seq2seq(self.valid_data)
            wandb.log({'valid-loss': valid_loss_}, step=epoch+1)
            
            print(f'{"-"*20} Epoch {epoch + 1}/{max_epoch} training done {"-"*20}')
            
            if valid_loss_ < best_valid_loss:
                best_valid_loss = valid_loss_
                self._save_model(self.model, model_dir, experiment_id)
                print(f'Save model in epoch {epoch + 1}.')
        
        print(f'{"-"*20} Retrieve best model for testing {"-"*20}')
        self._load_model(self.model, model_dir, experiment_id)
        self.model.to(self.device)
        valid_loss = self._evaluate_seq2seq(self.test_data)
        print(f'SEMI experiment done in {time.time() - EXP_START}s.')
        print(f'{"-"*40}')

        self.main_inference(self.test_data, self.test_split, interpolate_latent=False)

    def _batchify(self, batch):
        
        src_list, trg_list = [], []
        
        for sets in batch:
            src = torch.tensor([self.bos_id] + sets[0] + [self.eos_id], dtype=torch.int64)
            trg = torch.tensor([self.bos_id] + sets[1] + [self.eos_id], dtype=torch.int64)
            src_list.append(src)
            trg_list.append(trg)
        src_ = pad_sequence(src_list, batch_first=True, padding_value=self.pad_id)
        trg_ = pad_sequence(trg_list, batch_first=True, padding_value=self.pad_id)

        return src_.to(self.device), trg_.to(self.device)

    def _build_data(self, un_train_size, train_size, valid_size, test_size, data='quora'):
        
        assert un_train_size >= train_size
        assert data in ['quora', 'mscoco']
        if data == 'quora':
            assert un_train_size <= self.configs.quora_train_max
            lm_size = self.configs.quora_train_max
            sentence_pairs = process_quora(self.configs.quora_fp)
            sentence_pairs = shuffle(sentence_pairs, random_state=1234)
            train_valid_split, test_split = sentence_pairs[:-test_size], sentence_pairs[-test_size:]
        if data == 'mscoco':
            assert un_train_size <= self.configs.mscoco_train_max
            lm_size = self.configs.mscoco_train_max
            train_test_split = process_mscoco(self.configs.mscoco_fp_train)
            valid_split = process_mscoco(self.configs.mscoco_fp_test)
            # train_test_split = shuffle(train_test_split, random_state=1234)
            test_split = train_test_split[-test_size:]
            train_valid_split = valid_split + train_test_split[:-test_size]
            train_valid_split = shuffle(train_valid_split, random_state=1234)
        
        print(f'Calculate origional stats.')
        calculate_stats(train_valid_split)
        train_valid_idx, test_idx = normalise(train_valid_split, test_split, self.configs.max_len-2)
        print(f'Calculate normalised stats.')
        calculate_stats(train_valid_idx)
        print('Calculate bound for test data')
        test_pred = [s[0] for s in test_split]
        calculate_bound(test_pred, test_split, True, True)

        lm_idx = train_valid_idx[:lm_size]

        train_idx = train_valid_idx[:train_size]
        valid_idx = train_valid_idx[-valid_size:]

        train_valid_idx = shuffle(train_valid_idx, random_state=1234)
        un_train_idx = train_valid_idx[:un_train_size]
        un_train_idx = shuffle(un_train_idx, random_state=1234)

        if data == 'quora':
            lm_dl = DataLoader(lm_idx, batch_size=self.configs.qu_batch_size, shuffle=True, collate_fn=self._batchify)
            un_dl = DataLoader(un_train_idx, batch_size=self.configs.qu_batch_size, shuffle=True, collate_fn=self._batchify)
            train_dl = DataLoader(train_idx, batch_size=self.configs.qu_batch_size, shuffle=True, collate_fn=self._batchify)
            valid_dl = DataLoader(valid_idx, batch_size=self.configs.qu_batch_size, shuffle=False, collate_fn=self._batchify)
            test_dl = DataLoader(test_idx, batch_size=self.configs.qu_batch_size, shuffle=False, collate_fn=self._batchify)

        if data == 'mscoco':
            lm_dl = DataLoader(lm_idx, batch_size=self.configs.ms_batch_size, shuffle=True, collate_fn=self._batchify)
            un_dl = DataLoader(un_train_idx, batch_size=self.configs.ms_batch_size, shuffle=True, collate_fn=self._batchify)
            train_dl = DataLoader(train_idx, batch_size=self.configs.ms_batch_size, shuffle=True, collate_fn=self._batchify)
            valid_dl = DataLoader(valid_idx, batch_size=self.configs.ms_batch_size, shuffle=False, collate_fn=self._batchify)
            test_dl = DataLoader(test_idx, batch_size=self.configs.ms_batch_size, shuffle=False, collate_fn=self._batchify)

        return lm_dl, un_dl, train_dl, valid_dl, test_dl, test_split

    def _train_seq2seq(self, dataloader, optimizer, grad_clip):
        self.model.train()
        epoch_loss = 0
        start_time = time.time()

        log_inter = len(dataloader) // 3
        if log_inter == 0: log_inter = 1 
        for idx, (src, trg) in enumerate(tqdm(dataloader)):
            
            optimizer.zero_grad()
            
            trg_ = self.model.sequence_to_sequence(src, trg)
            loss_trg = self.model._reconstruction_loss(trg_, trg[:, 1:])
            loss = torch.mean(loss_trg) 
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.encoder.parameters(), grad_clip)
            torch.nn.utils.clip_grad_norm_(self.model.decoder.parameters(), grad_clip)

            optimizer.step()

            epoch_loss += loss.item()

            if idx % log_inter == 0 and idx > 0:    
                print(f'| Batches: {idx}/{len(dataloader)} | PPL: {math.exp(epoch_loss/(idx+1))} | LOSS: {epoch_loss/(idx+1)} |')
        
        elapsed = time.time() - start_time
        print(f'Epoch training time is: {elapsed}s.')
        return epoch_loss / len(dataloader)

    def _train_seq2seq_mtl(self, dataloader, optimizer, grad_clip):
        self.model.train()
        epoch_loss = 0
        start_time = time.time()

        log_inter = len(dataloader) // 3
        if log_inter == 0: log_inter = 1 
        for idx, (src, trg) in enumerate(tqdm(dataloader)):
            
            optimizer.zero_grad()
            
            src_, trg_ = self.model.sequence_to_sequence_mtl(src, trg)
            
            loss_src = self.model._reconstruction_loss(src_, src[:, 1:])
            loss_trg = self.model._reconstruction_loss(trg_, trg[:, 1:])
            
            loss = torch.mean(loss_trg) + torch.mean(loss_src)
            r_un = self.configs.un_train_size / (self.configs.un_train_size + self.configs.train_size)
            loss = loss * (1-r_un)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.encoder.parameters(), grad_clip)
            torch.nn.utils.clip_grad_norm_(self.model.decoder.parameters(), grad_clip)

            optimizer.step()

            epoch_loss += loss.item() / 2

            if idx % log_inter == 0 and idx > 0:    
                print(f'| Batches: {idx}/{len(dataloader)} | PPL: {math.exp(epoch_loss/(idx+1))} | LOSS: {epoch_loss/(idx+1)} |')
        
        elapsed = time.time() - start_time
        print(f'Epoch training time is: {elapsed}s.')
        return epoch_loss / len(dataloader)

    def _train_seq2seq_mtl2(self, dataloader, optimizer, grad_clip, temperature=None):
        self.model.train()
        epoch_loss = 0
        start_time = time.time()

        log_inter = len(dataloader) // 3
        if log_inter == 0: log_inter = 1 
        for idx, (src, trg) in enumerate(tqdm(dataloader)):
            
            optimizer.zero_grad()
            
            src_, trg_ = self.model.sequence_to_sequence_mtl(src, trg)
            
            loss_src = self.model._reconstruction_loss(src_, src[:, 1:])
            loss_trg = self.model._reconstruction_loss(trg_, trg[:, 1:])
            
            loss = torch.mean(loss_trg) + torch.mean(loss_src)
            r_un = self.configs.un_train_size / (self.configs.un_train_size + self.configs.train_size)
            loss = loss * (1-r_un)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.encoder.parameters(), grad_clip)
            torch.nn.utils.clip_grad_norm_(self.model.decoder.parameters(), grad_clip)

            optimizer.step()

            epoch_loss += loss.item() / 2

            if idx % log_inter == 0 and idx > 0:    
                print(f'| Batches: {idx}/{len(dataloader)} | PPL: {math.exp(epoch_loss/(idx+1))} | LOSS: {epoch_loss/(idx+1)} |')
        
        elapsed = time.time() - start_time
        print(f'Epoch training time is: {elapsed}s.')
        return epoch_loss / len(dataloader)
    
    def _train_ddl(self, dataloader, optimizer, grad_clip):
        self.model.train()
        epoch_loss = 0
        start_time = time.time()

        log_inter = len(dataloader) // 3
        if log_inter == 0: log_inter = 1 
        for idx, (src, trg) in enumerate(tqdm(dataloader)):
            
            optimizer.zero_grad()
            
            src_, trg_, recon_src, recon_trg = self.model.dual_directional_learning(src, trg)
            # src__, trg__ = self.model.supervised_reconstruction(src, trg, gumbel_max=self.configs.gumbel_max, temperature=temperature)
            # _, _, V = src_.size()
            loss_src = self.model._reconstruction_loss(src_, src[:, 1:])
            loss_trg = self.model._reconstruction_loss(trg_, trg[:, 1:])
            loss_src_recon = self.model._reconstruction_loss(recon_src, src[:, 1:])
            loss_trg_recon = self.model._reconstruction_loss(recon_trg, trg[:, 1:])

            # loss_src = F.nll_loss(torch.log(src_.reshape(-1, V)), src[:, 1:].reshape(-1), ignore_index=self.configs.pad_id, reduction='none')
            # loss_trg = F.nll_loss(torch.log(trg_.reshape(-1, V)), trg[:, 1:].reshape(-1), ignore_index=self.configs.pad_id, reduction='none')
            # loss_trg = self.model._reconstruction_loss(trg_, trg[:, 1:])

            # loss_src_ = self.model._reconstruction_loss(src__, src[:, 1:])
            # loss_trg_ = self.model._reconstruction_loss(trg__, trg[:, 1:])

            
            loss = torch.mean(loss_trg) + torch.mean(loss_src) + torch.mean(loss_src_recon) + torch.mean(loss_trg_recon)

            # loss = torch.mean(loss_trg) + torch.mean(loss_src) + torch.mean(loss_trg_) + torch.mean(loss_src_)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.encoder.parameters(), grad_clip)
            torch.nn.utils.clip_grad_norm_(self.model.decoder.parameters(), grad_clip)

            optimizer.step()

            epoch_loss += loss.item() / 4

            if idx % log_inter == 0 and idx > 0:    
                print(f'| Batches: {idx}/{len(dataloader)} | PPL: {math.exp(epoch_loss/(idx+1))} | LOSS: {epoch_loss/(idx+1)} |')
        
        elapsed = time.time() - start_time
        print(f'Epoch training time is: {elapsed}s.')
        return epoch_loss / len(dataloader)

    def _train_unsupervised(self, dataloader, optimizer, grad_clip, temperature, beta=1, top_k=500):
        self.model.train()
        epoch_total_loss, epoch_rec_loss, epoch_kl_loss = 0, 0, 0
        start_time = time.time()

        log_inter = len(dataloader) // 3
        if log_inter == 0: log_inter = 1 
        for idx, (src, _) in enumerate(tqdm(dataloader)):
            optimizer.zero_grad()
            _, q_trg, src_ = self.model.unsupervised_reconstruction(src, temperature, top_k)
            rec_loss = self.model._reconstruction_loss(src_, src[:, 1:])

            # _, S = src.size()
            # q_trg = q_trg[:, :(S-1)]
            

            if self.configs.use_lm:
                p_trg = self.model.language_modelling(src)
            else:
                p_trg = F.one_hot(src[:, 1:], self.configs.vocab_size).double().to(self.device)
                q_trg = F.softmax(q_trg, dim=2)
            
            # print(q_trg.size())
            # print(p_trg.size())

            kl_loss = self.model._KL_loss(q_trg, p_trg)

            loss = torch.mean(rec_loss) + beta * torch.mean(kl_loss)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.encoder.parameters(), grad_clip)
            torch.nn.utils.clip_grad_norm_(self.model.decoder.parameters(), grad_clip)

            optimizer.step()

            epoch_total_loss += loss.item()

            epoch_rec_loss += torch.mean(rec_loss).item()
            epoch_kl_loss += torch.mean(kl_loss).item()

            if idx % log_inter == 0 and idx > 0:
                print(f'{"-"*20} Batches: {idx}/{len(dataloader)} {"-"*20}')
                print(f'{"-"*20} Unsupervised Training Loss {"-"*20}')
                print(f'| TOTAL LOSS: {epoch_total_loss/(idx+1)} | word perplexity: {math.exp(epoch_rec_loss/(idx+1))} | REC Loss: {epoch_rec_loss/(idx+1)} | KL Loss: {epoch_kl_loss/(idx+1)} |')
        elapsed = time.time() - start_time
        print(f'Epoch training time is: {elapsed}s.')
        return epoch_total_loss / len(dataloader)
    
    def _evaluate_seq2seq(self, dataloader):
        self.model.eval()
        epoch_loss = 0
        start_time = time.time()
        for idx, (src, trg) in enumerate(tqdm(dataloader)):
            trg_ = self.model.sequence_to_sequence(src, trg)
            loss = self.model._reconstruction_loss(trg_, trg[:, 1:])
            loss = torch.mean(loss)
            epoch_loss += loss.item()
        print(f'{"-"*40}')
        print(f'| word perplexity: {math.exp(epoch_loss/(idx+1))} | loss: {epoch_loss/(idx+1)} |')
        print(f'Epoch evaluateion time is: {time.time() - start_time} s.')
        print(f'{"-"*40}')
        return epoch_loss / len(dataloader)
    
    def _train_lm(self, dataloader, optimizer, grad_clip):
        self.model.prior.train()
        epoch_loss = 0
        start_time = time.time()
        log_inter = len(dataloader) // 3
        if log_inter == 0: log_inter = 1 
        for idx, (src, _) in enumerate(tqdm(dataloader)):
            optimizer.zero_grad()          
            output = self.model.language_modelling(src)         
            loss = self.model._reconstruction_loss(output, src[:, 1:])           
            loss = torch.mean(loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.prior.parameters(), grad_clip)
            optimizer.step()
            epoch_loss += loss.item()
            if idx % log_inter == 0 and idx > 0:
                print(f'| Batches: {idx}/{len(dataloader)} | Running Loss: {epoch_loss/(idx+1)} | per word perplexity: {math.exp(epoch_loss/(idx+1))} |')
        elapsed = time.time() - start_time
        print(f'Epoch training time is: {elapsed}s.')
        return epoch_loss / len(dataloader)
    
    def _evaluate_lm(self, dataloader):
        self.model.prior.eval()
        epoch_loss = 0
        start_time = time.time()
        for idx, (src, _) in enumerate(tqdm(dataloader)):
            output = self.model.language_modelling(src)       
            loss = self.model._reconstruction_loss(output, src[:, 1:])
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