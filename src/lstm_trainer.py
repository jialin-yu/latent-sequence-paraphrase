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
from lstm import LSTM
from sample import straight_through_softmax

from pipeline import index_to_token, remove_bos_eos, stringify, bert_tokenizer
from sklearn.utils import shuffle, resample
import wandb


class Trainer(object):

    def __init__(self, configs):
        
        self.configs = configs
        self.configs.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = self.configs.device
        self.tokenizer = bert_tokenizer
        self.bos_id = self.tokenizer.bos_token_id
        self.eos_id = self.tokenizer.eos_token_id
        self.pad_id = self.tokenizer.pad_token_id

        assert self.configs.data in ['quora', 'mscoco']
        
        if self.configs.data == 'quora':
            lm_data, un_data, train_data, valid_data, test_data, test_split = self._build_quora_data(self.configs.un_train_size, 
            self.configs.train_size, self.configs.quora_valid, self.configs.quora_test)
        if self.configs.data == 'mscoco':
            lm_data, un_data, train_data, valid_data, test_data, test_split = self._build_mscoco_data(self.configs.un_train_size,
            self.configs.train_size, self.configs.mscoco_valid, self.configs.mscoco_test)
        
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

        self.model = LSTM(self.configs)
        self.model.to(self.device)
        print(f'{"-"*20} Model Description {"-"*20}')
        print(f'Set model as {self._count_parameters(self.model)} trainable parameters')
        print(f'{"-"*40}')
    
    def main_inference(self, dataloader, test_split, interpolate_latent=True):
        self.model.eval()

        print(f'{"-"*20} Perform inference {"-"*20}')

        pred_idx = []
        first_n = 40
        latent_samples = 5
        for idx, (src, trg) in enumerate(tqdm(dataloader)):      
            trg_ = self.model.inference(self.model.src_encoder, self.model.trg_decoder, src, self.configs.max_len)
            for trg in trg_:
                pred_idx.append(trg.cpu())

        if interpolate_latent:
            print(f'{"-"*20} Perform Latent Sampling {"-"*20}')
            latent_sample_idx = self._latent_interpolate(dataloader, first_n, latent_samples)
        
        test_ = [index_to_token(remove_bos_eos(pred)) for pred in pred_idx]
        
        if interpolate_latent:
            latent_sample_ = [[index_to_token(remove_bos_eos(s)) for s in sets] for sets in latent_sample_idx]
        
        if interpolate_latent:
            table = wandb.Table(columns=['id', 'pred', 'ref', 'ls'])
        else:
            table = wandb.Table(columns=['id', 'pred', 'ref'])

        print(f'{"-"*20} Calculate Final Results {"-"*20}')
        calculate_bound(test_, test_split, True, True, True)
        print(f'{"-"*20} Print first {first_n} Results {"-"*20}')
        test__ = test_[:first_n]
        if interpolate_latent:
            for index, _ in enumerate(test__):
                print(f'{"-"*20} Print {index + 1} Example {"-"*20}')
                print(f'Source: {stringify(test_split[index][0])}')
                print(f'Prediction: {stringify(test__[index])}')
                ref = test_split[index][1:]
                for reff in ref:
                    print(f'Reference: {stringify(reff)}')
                for latent in latent_sample_[index]:
                    print(f'Latent Sample: {stringify(latent)}')
                table.add_data(index, stringify(test__[index]), [stringify(reff) for reff in ref], [stringify(latent) for latent in latent_sample_[index]])
            print(f'{"-"*40}')
        else:
            for index, _ in enumerate(test__):
                print(f'{"-"*20} Print {index + 1} Example {"-"*20}')
                print(f'Source: {stringify(test_split[index][0])}')
                print(f'Prediction: {stringify(test__[index])}')
                ref = test_split[index][1:]
                for reff in ref:
                    print(f'Reference: {stringify(reff)}')
                table.add_data(index, stringify(test__[index]), [stringify(reff) for reff in ref])
            print(f'{"-"*40}')
        
        wandb.log({'Table': table})
        print(f'{"-"*20} Inference done {"-"*20}')
        
    def _latent_interpolate(self, dataloader, first_n, size):
        latent_sample_idx = []
        counter = 0

        for idx, (src, trg) in enumerate(tqdm(dataloader)):
                for index, _ in enumerate(src):
                    temp = []
                    for j in range(size):
                        _, latent_, _ = self.model.encode_sample_decode(self.model.src_encoder, self.model.trg_decoder, src[index].unsqueeze(0), self.configs.gumbel_max, 0.01)
                        temp.append(latent_.squeeze().cpu())
                    latent_sample_idx.append(temp)
                    counter += 1
                    if counter == first_n:
                        return latent_sample_idx
    
    def main_lm(self):
        wandb.init(project='paraphrase-semi', config=self.configs, entity='du_jialin')
        model_dir = self.configs.lm_dir
        max_epoch = self.configs.lm_max_epoch
        lr = self.configs.lm_lr
        grad_clip = self.configs.grad_clip
        seed = self.configs.seed
        experiment_id = self.configs.lm_id
        use_pseudo = self.configs.use_pseudo

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
    
    def main_seq2seq(self):
        wandb.init(project='paraphrase-seq2seq', config=self.configs, entity='du_jialin')
        model_dir = self.configs.seq2seq_dir
        max_epoch = self.configs.seq2seq_max_epoch
        lr = self.configs.seq2seq_lr
        grad_clip = self.configs.grad_clip
        seed = self.configs.seed
        experiment_id = self.configs.seq2seq_id
        seq2seq_duo = self.configs.duo

        print(f'{"-"*20} Initialise seq2seq experiment {"-"*20}')
        print(f'Use model directory {model_dir}')
        print(f'Experiment run for max {max_epoch} epochs')
        print(f'Learning rate set as {lr}')
        print(f'Use gradient clip of {grad_clip}')
        print(f'Use seed of {seed}')
        print(f'Experiment id as {experiment_id}')
        print(f'Total model parameters {self._count_parameters(self.model)-self._count_parameters(self.model.prior)} ')
        print(f'{"-"*40}')

        self._set_experiment_seed(seed)
        self.model.src_encoder.apply(self._xavier_initialize)
        self.model.trg_decoder.apply(self._xavier_initialize)
        self.model.trg_encoder.apply(self._xavier_initialize)
        self.model.src_decoder.apply(self._xavier_initialize)
        params = list(self.model.src_encoder.parameters()) + list(self.model.trg_decoder.parameters()) + list(self.model.trg_encoder.parameters()) + list(self.model.src_decoder.parameters())
        optimizer = torch.optim.Adam(params, lr=lr)
        
        EXP_START = time.time()
        best_valid_loss = float('inf')
        for epoch in range(max_epoch):
            
            train_loss = self._train_seq2seq(self.train_data, optimizer, grad_clip, duo=seq2seq_duo)
            valid_loss = self._evaluate_seq2seq(self.valid_data, duo=seq2seq_duo)
            
            if seq2seq_duo:
                wandb.log({'train-loss_duo': train_loss}, step=epoch+1)
                wandb.log({'valid-loss_duo': valid_loss}, step=epoch+1)
            else:
                wandb.log({'train-loss': train_loss}, step=epoch+1)
                wandb.log({'valid-loss': valid_loss}, step=epoch+1)
            
            print(f'{"-"*20} Epoch {epoch + 1}/{max_epoch} training done {"-"*20}')
            
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss 
                self._save_model(self.model, model_dir, experiment_id)
                print(f'Save model in epoch {epoch + 1}.')
        
        
        print(f'{"-"*20} Retrieve best model for testing {"-"*20}')
        self._load_model(self.model, model_dir, experiment_id)
        self.model.to(self.device)
        valid_loss = self._evaluate_seq2seq(self.test_data, duo=seq2seq_duo)
        print(f'Seq2seq experiment done in {time.time() - EXP_START}s.')
        print(f'{"-"*40}')

        self.main_inference(self.test_data, self.test_split, interpolate_latent=False)

    def main_vae(self):
        lm_dir = self.configs.lm_dir
        lm_id = self.configs.lm_id

        model_dir = self.configs.vae_dir
        max_epoch = self.configs.vae_max_epoch
        lr = self.configs.vae_lr
        grad_clip = self.configs.grad_clip
        seed = self.configs.seed
        experiment_id = self.configs.vae_id

        if self.configs.use_pretrain_lm:
            self._load_model(self.model, lm_dir, lm_id)
        
        beta_factor = 1
        # temperature guildine from https://sassafras13.github.io/GumbelSoftmax/
        if self.configs.fixed_temperature:
            high_temp = 0.1
            low_temp = 0.1
        else:
            high_temp = 10
            low_temp = 0.01
        temperature = list(np.linspace(high_temp, low_temp, max_epoch))

        print(f'{"-"*20} Initialise VAE experiment {"-"*20}')
        print(f'Use model directory {model_dir}')
        print(f'Experiment run for max {max_epoch} epochs')
        print(f'Learning rate set as {lr}')
        print(f'Use gradient clip of {grad_clip}')
        print(f'Use seed of {seed}')
        print(f'Experiment id as {experiment_id}')
        
        if self.configs.use_pretrain_lm:
            print(f'Use pre-trained LM from dir {lm_dir} and id {lm_id}')
        else:
            print(f'No pre-trained LM used')

        print(f'Total model parameters {self._count_parameters(self.model)-self._count_parameters(self.model.prior)} ')
        print(f'Set high temperate as {high_temp} amd low temperature as {low_temp}')
        print(f'Training with beta factor of {beta_factor}')
        print(f'{"-"*40}')

        
        self._set_experiment_seed(seed)
        self.model.src_encoder.apply(self._xavier_initialize)
        self.model.trg_decoder.apply(self._xavier_initialize)
        self.model.trg_encoder.apply(self._xavier_initialize)
        self.model.src_decoder.apply(self._xavier_initialize)
        params = list(self.model.src_encoder.parameters()) + list(self.model.trg_decoder.parameters()) + list(self.model.trg_encoder.parameters()) + list(self.model.src_decoder.parameters())
        optimizer = torch.optim.Adam(params, lr=lr)
        
        EXP_START = time.time()
        best_valid_loss = float('inf')
        for epoch in range(max_epoch):
            train_loss = self._train_vae(self.un_data, optimizer, grad_clip, temperature[epoch], beta_factor)
            valid_loss = self._evaluate_vae(self.valid_data, temperature[epoch], beta_factor)
            print(f'{"-"*20} Epoch {epoch + 1}/{max_epoch} training done {"-"*20}') 
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss 
                self._save_model(self.model, model_dir, experiment_id)
                print(f'Save model in epoch {epoch + 1}.')
        
        print(f'{"-"*20} Retrieve best model for testing {"-"*20}')
        self._load_model(self.model, model_dir, experiment_id)
        self.model.to(self.device)
        valid_loss = self._evaluate_vae(self.test_data, low_temp)
        print(f'VAE experiment done in {time.time() - EXP_START}s.')

    def main_semi_supervised(self):

        wandb.init(project='paraphrase-semi', config=self.configs, entity='du_jialin')
        lm_dir = self.configs.lm_dir
        lm_id = self.configs.lm_id

        model_dir = self.configs.semi_dir
        max_epoch = self.configs.semi_max_epoch
        lr = self.configs.semi_lr
        grad_clip = self.configs.grad_clip
        seed = self.configs.seed
        experiment_id = self.configs.semi_id
        
        if self.configs.use_pretrain_lm:
            self._load_model(self.model, lm_dir, lm_id)

        # https://iancovert.com/blog/concrete_temperature/
        # if fixed temperature, set to 0.1 as in http://proceedings.mlr.press/v80/chen18j/chen18j.pdf
        # else, set high to 10, low to 0.01 as in http://proceedings.mlr.press/v97/balin19a/balin19a.pdf
        if self.configs.fixed_temperature:
            high_temp = 0.1
            low_temp = 0.1
        else:
            high_temp = 10
            low_temp = 0.01
        # if self.configs.un_train_size == self.configs.train_size:
        #     # alpha_factor = self.configs.un_train_size / (self.configs.un_train_size + self.configs.train_size)
        #     alpha_factor = 0.5
        # else:
        #     alpha_factor = self.configs.un_train_size / (self.configs.un_train_size + self.configs.train_size)
        
        beta_factor = 0.5
        temperature = list(np.linspace(high_temp, low_temp, max_epoch))

        print(f'{"-"*20} Initialise SEMI experiment {"-"*20}')
        print(f'Use model directory {model_dir}')
        print(f'Experiment run for max {max_epoch} epochs')
        print(f'Learning rate set as {lr}')
        print(f'Use gradient clip of {grad_clip}')
        print(f'Use seed of {seed}')
        print(f'Experiment id as {experiment_id}')
        
        if self.configs.use_pretrain_lm:
            print(f'Use pre-trained LM from dir {lm_dir} and id {lm_id}')
        else:
            print(f'No pre-trained LM used')

        print(f'Total model parameters {self._count_parameters(self.model)-self._count_parameters(self.model.prior)} ')
        print(f'Set high temperate as {high_temp} amd low temperature as {low_temp}')
        print(f'Training with beta factor of {beta_factor} for KL')
        print(f'{"-"*40}')

        
        self._set_experiment_seed(seed)
        self.model.src_encoder.apply(self._xavier_initialize)
        self.model.trg_decoder.apply(self._xavier_initialize)
        self.model.trg_encoder.apply(self._xavier_initialize)
        self.model.src_decoder.apply(self._xavier_initialize)
        params = list(self.model.src_encoder.parameters()) + list(self.model.trg_decoder.parameters()) + list(self.model.trg_encoder.parameters()) + list(self.model.src_decoder.parameters())
        optimizer = torch.optim.Adam(params, lr=lr)
        
        EXP_START = time.time()
        best_valid_loss = float('inf')
        for epoch in range(max_epoch):
            if self.configs.un_train_size == self.configs.train_size:
                train_loss = self._train_semi(self.train_data, optimizer, grad_clip, temperature[epoch], beta_factor, balanced=True)
                wandb.log({'train-loss': train_loss}, step=epoch+1)
            else:
                train_loss_b = self._train_semi(self.train_data, optimizer, grad_clip, temperature[epoch], beta_factor, balanced=True)
                train_loss_nb = self._train_semi(self.un_data, optimizer, grad_clip, temperature[epoch], beta_factor, balanced=False)
                train_loss = train_loss_b + train_loss_nb
                wandb.log({'train-loss': train_loss}, step=epoch+1)

            valid_loss = self._evaluate_semi(self.valid_data, low_temp, beta_factor)
            wandb.log({'valid-loss': valid_loss}, step=epoch+1)
            valid_loss_ = self._evaluate_seq2seq(self.valid_data, duo=True)
            wandb.log({'valid-seq2seq-loss': valid_loss_}, step=epoch+1)
            
            print(f'{"-"*20} Epoch {epoch + 1}/{max_epoch} training done {"-"*20}')
            
            if valid_loss_ < best_valid_loss:
                best_valid_loss = valid_loss_
                self._save_model(self.model, model_dir, experiment_id)
                print(f'Save model in epoch {epoch + 1}.')
        
        print(f'{"-"*20} Retrieve best model for testing {"-"*20}')
        self._load_model(self.model, model_dir, experiment_id)
        self.model.to(self.device)
        valid_loss = self._evaluate_semi(self.test_data, low_temp, beta_factor)
        print(f'SEMI experiment done in {time.time() - EXP_START}s.')
        print(f'{"-"*40}')

        self.main_inference(self.test_data, self.test_split, interpolate_latent=True)

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

    def _build_quora_data(self, un_train_size, train_size, valid_size, test_size):
        
        assert un_train_size >= train_size
        assert un_train_size <= self.configs.quora_train_max
        lm_size = self.configs.quora_train_max
        
        sentence_pairs = process_quora(self.configs.quora_fp)
        sentence_pairs = shuffle(sentence_pairs, random_state=1234)
        train_valid_split, test_split = sentence_pairs[:-test_size], sentence_pairs[-test_size:]
        print(f'Calculate origional stats.')
        calculate_stats(train_valid_split)
        train_valid_idx, test_idx = normalise(train_valid_split, test_split, self.configs.quora_max_len)
        print(f'Calculate normalised stats.')
        calculate_stats(train_valid_idx)
        print('Calculate bound for test data')
        test_pred = [s[0] for s in test_split]
        calculate_bound(test_pred, test_split, True, True)

        train_valid_idx = shuffle(train_valid_idx, random_state=1234)
        lm_idx = train_valid_idx[:lm_size]
        lm_idx = shuffle(lm_idx, random_state=1234)

        train_idx = train_valid_idx[:train_size]
        # train_idx = train_idx + resample(train_idx, n_samples=len_diff, random_state=1234)
        train_idx = shuffle(train_idx, random_state=1234)

        if un_train_size > train_size:
            un_train_idx = train_valid_idx[train_size:un_train_size]
            un_train_idx = shuffle(un_train_idx, random_state=1234)
        else:
            # not gonna use this 
            un_train_idx = train_idx[:1000]

        valid_idx = train_valid_idx[-valid_size:]

        lm_dl = DataLoader(lm_idx, batch_size=self.configs.qu_batch_size, shuffle=True, collate_fn=self._batchify)
        un_dl = DataLoader(un_train_idx, batch_size=self.configs.qu_batch_size, shuffle=True, collate_fn=self._batchify)
        train_dl = DataLoader(train_idx, batch_size=self.configs.qu_batch_size, shuffle=True, collate_fn=self._batchify)
        valid_dl = DataLoader(valid_idx, batch_size=self.configs.qu_batch_size, shuffle=False, collate_fn=self._batchify)
        test_dl = DataLoader(test_idx, batch_size=self.configs.qu_batch_size, shuffle=False, collate_fn=self._batchify)

        return lm_dl, un_dl, train_dl, valid_dl, test_dl, test_split

    def _build_mscoco_data(self, un_train_size, train_size, valid_size, test_size):
        
        assert un_train_size >= train_size
        assert un_train_size <= self.configs.mscoco_train_max
        len_diff = un_train_size - train_size
        lm_size = self.configs.mscoco_train_max
        
        train_valid_split = process_mscoco(self.configs.mscoco_fp_train)
        test_split = process_mscoco(self.configs.mscoco_fp_test)
        test_split = shuffle(test_split, random_state=1234)
        test_split = test_split[-test_size:]
        
        print(f'Calculate origional stats.')
        calculate_stats(train_valid_split)
        train_valid_idx, test_idx = normalise(train_valid_split, test_split, self.configs.mscoco_max_len)
        print(f'Calculate normalised stats.')
        calculate_stats(train_valid_idx)
        print('Calculate bound for test data')
        test_pred = [s[0] for s in test_split]
        calculate_bound(test_pred, test_split, True, True)

        train_valid_idx = shuffle(train_valid_idx, random_state=1234)
        lm_idx = train_valid_idx[:lm_size]
        lm_idx = shuffle(lm_idx, random_state=1234)

        train_idx = train_valid_idx[:train_size]
        # train_idx = train_idx + resample(train_idx, n_samples=len_diff, random_state=1234)
        train_idx = shuffle(train_idx, random_state=1234)

        if un_train_size > train_size:
            un_train_idx = train_valid_idx[train_size:un_train_size]
            un_train_idx = shuffle(un_train_idx, random_state=1234)
        else:
            # not gonna use this 
            un_train_idx = train_idx[:1000]

        valid_idx = train_valid_idx[-valid_size:]

        lm_dl = DataLoader(lm_idx, batch_size=self.configs.qu_batch_size, shuffle=True, collate_fn=self._batchify)
        un_dl = DataLoader(un_train_idx, batch_size=self.configs.ms_batch_size, shuffle=True, collate_fn=self._batchify)
        train_dl = DataLoader(train_idx, batch_size=self.configs.ms_batch_size, shuffle=True, collate_fn=self._batchify)
        valid_dl = DataLoader(valid_idx, batch_size=self.configs.ms_batch_size, shuffle=False, collate_fn=self._batchify)
        test_dl = DataLoader(test_idx, batch_size=self.configs.ms_batch_size, shuffle=False, collate_fn=self._batchify)

        return lm_dl, un_dl, train_dl, valid_dl, test_dl, test_split

    def _train_lm(self, dataloader, optimizer, grad_clip):
        self.model.prior.train()
        epoch_loss = 0
        start_time = time.time()
        
        log_inter = len(dataloader) // 5
        for idx, (src, trg) in enumerate(tqdm(dataloader)):
            optimizer.zero_grad()          
                    
            output = self.model.decode_lm(self.model.prior, src[:, :-1])         
            loss = self.model._reconstruction_loss(output, src[:, 1:])

            
            loss = torch.mean(loss)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.prior.parameters(), grad_clip)
            optimizer.step()

            epoch_loss += loss.item()
            if idx % log_inter == 0 and idx > 0:
                print(f'| Batches: {idx}/{len(dataloader)} | Running Loss: {epoch_loss/(idx+1)} | PPL: {math.exp(epoch_loss/(idx+1))} |')
        elapsed = time.time() - start_time
        print(f'Epoch training time is: {elapsed}s.')
        return epoch_loss / len(dataloader)
    
    def _evaluate_lm(self, dataloader):
        self.model.prior.eval()
        epoch_loss = 0
        start_time = time.time()
        for idx, (src, trg) in enumerate(tqdm(dataloader)):
            


            output = self.model.decode_lm(self.model.prior, src[:, :-1])         
            loss = self.model._reconstruction_loss(output, src[:, 1:])
                
            batch_loss = torch.mean(loss)
            epoch_loss += batch_loss.item()
            
        print(f'| Total Loss: {epoch_loss/(idx+1)} | PPL: {math.exp(epoch_loss/(idx+1))} |')
        elapsed = time.time() - start_time
        print(f'Epoch Evaluation time is: {elapsed}s.')
        return epoch_loss / len(dataloader)
    
    def _train_seq2seq(self, dataloader, optimizer, grad_clip, duo=True):
        self.model.train()
        epoch_loss = 0
        start_time = time.time()

        log_inter = len(dataloader) // 3
        for idx, (src, trg) in enumerate(tqdm(dataloader)):
            optimizer.zero_grad()
            
            src_ = self.model.encode_and_decode(self.model.trg_encoder, self.model.src_decoder,
                trg, src[:,:-1])
            trg_ = self.model.encode_and_decode(self.model.src_encoder, self.model.trg_decoder,
                src, trg[:,:-1])

            loss_src = self.model._reconstruction_loss(src_, src[:, 1:])
            loss_trg = self.model._reconstruction_loss(trg_, trg[:, 1:])

            if duo:
                loss = torch.mean(loss_src) + torch.mean(loss_trg)
                loss.backward()
            else:
                loss = torch.mean(loss_trg)
                loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.src_encoder.parameters(), grad_clip)
            torch.nn.utils.clip_grad_norm_(self.model.trg_decoder.parameters(), grad_clip)
            torch.nn.utils.clip_grad_norm_(self.model.trg_encoder.parameters(), grad_clip)
            torch.nn.utils.clip_grad_norm_(self.model.src_decoder.parameters(), grad_clip)

            optimizer.step()

            if duo:
                epoch_loss += (loss.item())/2
            else:
                epoch_loss += loss.item()

            if idx % log_inter == 0 and idx > 0:    
                print(f'| Batches: {idx}/{len(dataloader)} | PPL: {math.exp(epoch_loss/(idx+1))} | LOSS: {epoch_loss/(idx+1)} |')
        
        elapsed = time.time() - start_time
        print(f'Epoch training time is: {elapsed}s.')
        return epoch_loss / len(dataloader)

    def _evaluate_seq2seq(self, dataloader, duo=True):
        self.model.eval()
        epoch_loss = 0
        start_time = time.time()
        
        for idx, (src, trg) in enumerate(tqdm(dataloader)):
            
            src_ = self.model.encode_and_decode(self.model.trg_encoder, self.model.src_decoder,
                trg, src[:,:-1])
            trg_ = self.model.encode_and_decode(self.model.src_encoder, self.model.trg_decoder,
                src, trg[:,:-1])

            loss_src = self.model._reconstruction_loss(src_, src[:, 1:])
            loss_trg = self.model._reconstruction_loss(trg_, trg[:, 1:])

            if duo:
                loss = torch.mean(loss_src) + torch.mean(loss_trg)
                epoch_loss += (loss.item()) / 2
            else:
                loss = torch.mean(loss_trg)
                epoch_loss += loss.item()

                 
        print(f'| PPL: {math.exp(epoch_loss/(idx+1))} | LOSS: {epoch_loss/(idx+1)} |')
        
        elapsed = time.time() - start_time
        print(f'Epoch evaluateion time is: {elapsed}s.')
        return epoch_loss / len(dataloader)

    def _train_semi(self, dataloader, optimizer, grad_clip, temperature=None, beta=0.5, balanced=True):
        self.model.train()
        epoch_total_loss, epoch_seq2seq_loss, epoch_vae_loss, epoch_rec_loss, epoch_kl_loss = 0, 0, 0, 0, 0
        start_time = time.time()
        
        log_inter = len(dataloader) // 5
        for idx, (src, trg) in enumerate(tqdm(dataloader)):
            optimizer.zero_grad()

            # trg_ (B, S, V)
            # trg_, trg_logit = self.model.encode_sample_decode(self.model.src_encoder, self.model.trg_decoder, 
                        # src, self.configs.gumbel_max, temperature) 

            trg_, trg_hard, trg_logit = self.model.encode_sample_decode(self.model.src_encoder, 
                self.model.trg_decoder, src, self.configs.gumbel_max, temperature) 

            # src__ (B, S-1, V)
            src__ = self.model.encode_and_decode(self.model.trg_encoder, self.model.src_decoder,
                trg_, src[:,:-1])

            rec_loss = self.model._reconstruction_loss(src__, src[:, 1:])
            
            # calculate reconstruction loss between src and src__
                
            if self.configs.use_pretrain_lm:

                p_trg = self.model.decode_lm(self.model.prior, src[:, :-1])
                p_trg = F.softmax(p_trg, dim=2)

            else:

                p_trg = F.one_hot(src[:, 1:], self.configs.vocab_size).double()

            q_trg = trg_logit
            
            kl_loss = self.model._KL_loss(q_trg, p_trg)

            vae_loss = torch.mean(rec_loss) + beta * torch.mean(kl_loss)

            src_de = self.model.encode_and_decode(self.model.trg_encoder, self.model.src_decoder,
                trg, src[:,:-1])
            
            trg_de = self.model.encode_and_decode(self.model.src_encoder, self.model.trg_decoder,
                src, trg[:,:-1])

            loss_src = self.model._reconstruction_loss(src_de, src[:, 1:])
            loss_trg = self.model._reconstruction_loss(trg_de, trg[:, 1:])

            seq2seq_loss = torch.mean(loss_src) + torch.mean(loss_trg)

            if balanced:
                loss = vae_loss + seq2seq_loss
            else:
                loss = vae_loss

            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.src_encoder.parameters(), grad_clip)
            torch.nn.utils.clip_grad_norm_(self.model.trg_decoder.parameters(), grad_clip)
            torch.nn.utils.clip_grad_norm_(self.model.trg_encoder.parameters(), grad_clip)
            torch.nn.utils.clip_grad_norm_(self.model.src_decoder.parameters(), grad_clip)

            optimizer.step()

            epoch_total_loss += loss.item()
            epoch_vae_loss += vae_loss.item()
            if balanced:
                epoch_seq2seq_loss += seq2seq_loss.item() / 2
            epoch_rec_loss += torch.mean(rec_loss).item()
            epoch_kl_loss += torch.mean(kl_loss).item()

            if idx % log_inter == 0 and idx > 0:
                print(f'{"-"*20} Batches: {idx}/{len(dataloader)} {"-"*20}')
                print(f'For VAE | TOTAL LOSS: {epoch_vae_loss/(idx+1)} | PPL: {math.exp(epoch_rec_loss/(idx+1))} | REC Loss: {epoch_rec_loss/(idx+1)} | KL Loss: {epoch_kl_loss/(idx+1)} |')
                if balanced:
                    print(f'| FOR SEQ2SEQ | TOTAL LOSS: {epoch_seq2seq_loss/(idx+1)} | PPL: {math.exp(epoch_seq2seq_loss/(idx+1))} |')
                print(f'| IN GENERAL | TOTAL LOSS: {epoch_total_loss/(idx+1)} |')
        elapsed = time.time() - start_time
        print(f'Epoch training time is: {elapsed}s.')
        return epoch_total_loss / len(dataloader)

    def _evaluate_semi(self, dataloader, temperature=None, beta=0.5):
        self.model.eval()
        epoch_total_loss, epoch_seq2seq_loss, epoch_vae_loss, epoch_rec_loss, epoch_kl_loss = 0, 0, 0, 0, 0
        start_time = time.time()
        
        for idx, (src, trg) in enumerate(tqdm(dataloader)):

            # trg_ (B, S, V)
            trg_, trg_hard, trg_logit = self.model.encode_sample_decode(self.model.src_encoder, 
                self.model.trg_decoder, src, self.configs.gumbel_max, temperature) 
            
            # src__ (B, S-1, V)
            src__ = self.model.encode_and_decode(self.model.trg_encoder, self.model.src_decoder,
                trg_, src[:,:-1])

            rec_loss = self.model._reconstruction_loss(src__, src[:, 1:])
            
            # calculate reconstruction loss between src and src__
                
            if self.configs.use_pretrain_lm:
                    
                p_trg = self.model.decode_lm(self.model.prior, src[:, :-1])
                p_trg = F.softmax(p_trg, dim=2)

            else:
                    
                p_trg = F.one_hot(src[:, 1:], self.configs.vocab_size).double()

            q_trg = trg_logit
            
            kl_loss = self.model._KL_loss(q_trg, p_trg)

            vae_loss = torch.mean(rec_loss) + beta * torch.mean(kl_loss)

            src_de = self.model.encode_and_decode(self.model.trg_encoder, self.model.src_decoder,
                trg, src[:,:-1])
            
            trg_de = self.model.encode_and_decode(self.model.src_encoder, self.model.trg_decoder,
                src, trg[:,:-1])

            loss_src = self.model._reconstruction_loss(src_de, src[:, 1:])
            loss_trg = self.model._reconstruction_loss(trg_de, trg[:, 1:])

            seq2seq_loss = torch.mean(loss_src) + torch.mean(loss_trg)

            loss = vae_loss + seq2seq_loss

            epoch_total_loss += loss.item()
            epoch_vae_loss += vae_loss.item()
            epoch_seq2seq_loss += seq2seq_loss.item() / 2
            epoch_rec_loss += torch.mean(rec_loss).item()
            epoch_kl_loss += torch.mean(kl_loss).item()

        print(f'{"-"*20} Evaluation Result {"-"*20}')
        print(f'For VAE | TOTAL LOSS: {epoch_vae_loss/(idx+1)} | PPL: {math.exp(epoch_rec_loss/(idx+1))} | REC Loss: {epoch_rec_loss/(idx+1)} | KL Loss: {epoch_kl_loss/(idx+1)} |')
        print(f'| FOR SEQ2SEQ | TOTAL LOSS: {epoch_seq2seq_loss/(idx+1)} | PPL: {math.exp(epoch_seq2seq_loss/(idx+1))} |')
        print(f'| IN GENERAL | TOTAL LOSS: {epoch_total_loss/(idx+1)} |')
        
        elapsed = time.time() - start_time
        print(f'Epoch training time is: {elapsed}s.')
        return epoch_total_loss / len(dataloader)

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