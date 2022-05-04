from numpy import argmax
import torch
import torch.nn as nn
import torch.nn.functional as F

from encoder import TransformerEncoder
from decoder import TransformerDecoder

from lm import LanguageModel

from sample import gumbel_softmax_topk, softmax_topk
    

class Transformer(nn.Module):

    def __init__(self, configs):
        super(Transformer, self).__init__()
        
        self.device = configs.device
        self.pad_id = configs.pad_id
        self.vocab_size = configs.vocab_size
       
        
        self.prior = LanguageModel(configs)

        self.encoder = TransformerEncoder(configs)
        self.decoder = TransformerDecoder(configs)
        
        self.loss = nn.CrossEntropyLoss(ignore_index=self.pad_id, reduction="none")
        self.nll_loss = nn.NLLLoss(ignore_index=self.pad_id, reduction="none")

        self.max_len = configs.max_len

    def _get_causal_mask(self, src, trg):
        '''
        Generate Causal Mask
        
        INPUT:
        src (B, S)
        trg (B, T)

        OUTPUT:
        [ x1 x2 x3 ] ==> [ 0 -inf -inf ]
                         [ 0   0  -inf ]
                         [ 0   0    0  ]

        src_m (S, S)
        trg_m (T, T)
        trg_src_m (T, S)
        '''

        _, S = src.size()
        _, T = trg.size()

        src_m = torch.triu(torch.ones(S, S) * float('-inf'), diagonal=1).to(self.device)
        trg_m = torch.triu(torch.ones(T, T) * float('-inf'), diagonal=1).to(self.device)
        trg_src_m = torch.triu(torch.ones(T, S) * float('-inf'), diagonal=1).to(self.device)

        return src_m, trg_m, trg_src_m

    def _get_padding_mask(self, src, trg):
        '''
        Generate Padding Mask

        INPUT:
        src (B, S)
        trg (B, T)

        OUTPUT: 
        [ x1 x2 x3 <pad> <pad> ] ==> [ 0 0 0 1 1 ]
        src_pm (B, S)
        trg_pm (B, T)
        '''
        _, S = src.size()
        _, T = trg.size()

        src_pm = (src == self.pad_id).to(self.device)
        trg_pm = (trg == self.pad_id).to(self.device)

        return src_pm, trg_pm

    def unsupervised_reconstruction_(self, src_idx, temperature, top_k):
        '''
        Unsupervised reconstruction with gumbel softmax sampling 
        https://arxiv.org/pdf/1611.01144.pdf
        https://arxiv.org/pdf/1611.00712.pdf
        
        INPUT:
        src_idx (B, S); <bos> x <eos>
        temperature: float number
        RETURN: 
        trg_idx_ (B, S) <bos> y_ <eos>
        trg_logits (B, S-1, V) y_ <eos>
        src_logits (B, S-1, V) x_ <eos>
        '''

        ###############################################
        # first generate pseudo teacher forcing labels
        ###############################################

        _, T = src_idx.size()
        src_pm, _ = self._get_padding_mask(src_idx, src_idx)
        enc_src = self.encoder.encode(src_idx, None, src_pm)
        trg_idx_ = src_idx[:, 0].unsqueeze(1).detach() # B, 1

        for _ in range(T-1):
            _, trg_m, trg_src_m = self._get_causal_mask(src_idx, trg_idx_)
            _, trg_pm = self._get_padding_mask(src_idx, trg_idx_)
            trg_out = self.decoder.decode(trg_idx_, enc_src, trg_m, trg_src_m, trg_pm, src_pm)
            trg_new_idx = torch.argmax(trg_out[:, -1], dim=1).unsqueeze(1)
            trg_idx_ = torch.cat((trg_idx_, trg_new_idx), dim=1).detach()

        final_out = trg_out.detach()
        # detach this now everything from teacher force is not associated with gradient
        tf_trg_ = gumbel_softmax_topk(final_out, temperature, top_k, True) 

        tf_trg_idx= torch.argmax(tf_trg_, dim=2)
        _, trg_m, trg_src_m = self._get_causal_mask(src_idx, tf_trg_idx)
        _, trg_pm = self._get_padding_mask(src_idx, tf_trg_idx)
        dec_out_trg = self.decoder.decode(tf_trg_, enc_src, trg_m, trg_src_m, trg_pm, src_pm)

        ###############################################
        # then perform sampling based on pseudo labels
        ###############################################

        out_trg_gumbel = gumbel_softmax_topk(dec_out_trg, temperature, top_k, True)
        trg_input = torch.cat((F.one_hot(src_idx[:, 0].unsqueeze(1), self.vocab_size).detach(), out_trg_gumbel), dim=1)
        trg_idx_ = torch.argmax(trg_input, dim=2)
        q_trg = out_trg_gumbel

        ###############################################
        # final reconstruct to original labels
        ###############################################

        src_pm_n, trg_pm_n = self._get_padding_mask(trg_idx_, src_idx[:, :-1])
        _, trg_m_n, trg_src_m_n = self._get_causal_mask(trg_idx_, src_idx[:, :-1])

        enc_src_n = self.encoder.encode(trg_input, None, src_pm_n)
        dec_out_src = self.decoder.decode(src_idx[:, :-1], enc_src_n, trg_m_n, trg_src_m_n, trg_pm_n, src_pm_n)

        return trg_idx_, q_trg, dec_out_src

    def unsupervised_reconstruction(self, src_idx, temperature, top_k):
        '''
        Unsupervised reconstruction with gumbel softmax sampling 
        https://arxiv.org/pdf/1611.01144.pdf
        https://arxiv.org/pdf/1611.00712.pdf
        
        INPUT:
        src_idx (B, S); <bos> x <eos>
        temperature: float number
        RETURN: 
        trg_idx_ (B, S) <bos> y_ <eos>
        trg_logits (B, S-1, V) y_ <eos>
        src_logits (B, S-1, V) x_ <eos>
        '''

        ###############################################
        # first generate pseudo teacher forcing labels
        ###############################################

        _, T = src_idx.size()
        src_pm, _ = self._get_padding_mask(src_idx, src_idx)
        enc_src = self.encoder.encode(src_idx, None, src_pm)
        trg_idx_ = src_idx[:, 0].unsqueeze(1).detach() # B, 1

        for _ in range(T-1):
            _, trg_m, trg_src_m = self._get_causal_mask(src_idx, trg_idx_)
            _, trg_pm = self._get_padding_mask(src_idx, trg_idx_)
            trg_out = self.decoder.decode(trg_idx_, enc_src, trg_m, trg_src_m, trg_pm, src_pm)
            trg_new_idx = torch.argmax(trg_out[:, -1], dim=1).unsqueeze(1)
            trg_idx_ = torch.cat((trg_idx_, trg_new_idx), dim=1).detach()

        tf_trg_idx = trg_idx_[:, :-1]
        # final_out = trg_out.detach()
        # detach this now everything from teacher force is not associated with gradient
        # tf_trg_ = gumbel_softmax_topk(final_out, temperature, top_k, True) 

        # tf_trg_idx= torch.argmax(tf_trg_, dim=2)
        _, trg_m, trg_src_m = self._get_causal_mask(src_idx, tf_trg_idx)
        _, trg_pm = self._get_padding_mask(src_idx, tf_trg_idx)
        dec_out_trg = self.decoder.decode(tf_trg_idx, enc_src, trg_m, trg_src_m, trg_pm, src_pm)

        ###############################################
        # then perform sampling based on pseudo labels
        ###############################################

        out_trg_gumbel = gumbel_softmax_topk(dec_out_trg, temperature, top_k, True)
        trg_input = torch.cat((F.one_hot(src_idx[:, 0].unsqueeze(1), self.vocab_size).detach(), out_trg_gumbel), dim=1)
        trg_idx_ = torch.argmax(trg_input, dim=2)
        q_trg = dec_out_trg

        ###############################################
        # final reconstruct to original labels
        ###############################################

        src_pm_n, trg_pm_n = self._get_padding_mask(trg_idx_, src_idx[:, :-1])
        _, trg_m_n, trg_src_m_n = self._get_causal_mask(trg_idx_, src_idx[:, :-1])

        enc_src_n = self.encoder.encode(trg_input, None, src_pm_n)
        dec_out_src = self.decoder.decode(src_idx[:, :-1], enc_src_n, trg_m_n, trg_src_m_n, trg_pm_n, src_pm_n)

        return trg_idx_, q_trg, dec_out_src
    
    def dual_directional_learning(self, src_idx, trg_idx):
        '''
        INPUT:
        src_idx (B, S); <bos> x <eos>
        trg_idx (B, T); <bos> y <eos>

        RETURN: 
        reconstruct_src (B, S-1, V) x_ <eos>
        reconstruct_trg (B, T-1, V) y_ <eos>
        '''

        ###############################################
        # first src to trg
        ###############################################
        
        # sequence to sequence learning

        src_pm, trg_pm = self._get_padding_mask(src_idx, trg_idx[:, :-1])
        _, trg_m, trg_src_m = self._get_causal_mask(src_idx, trg_idx[:, :-1])
            
        enc_src = self.encoder.encode(src_idx, None, src_pm)
        dec_out_trg = self.decoder.decode(trg_idx[:, :-1], enc_src, trg_m, trg_src_m, trg_pm, src_pm)

        ###############################################
        # then trg to src
        ###############################################

        src_pm_n, trg_pm_n = self._get_padding_mask(trg_idx, src_idx[:, :-1])
        _, trg_m_n, trg_src_m_n = self._get_causal_mask(trg_idx, src_idx[:, :-1])
            
        enc_src_n = self.encoder.encode(trg_idx, None, src_pm_n)
        dec_out_src = self.decoder.decode(src_idx[:, :-1], enc_src_n, trg_m_n, trg_src_m_n, trg_pm_n, src_pm_n)

        return dec_out_src, dec_out_trg, dec_out_src, dec_out_trg
    
    def dual_directional_learning_(self, src_idx, trg_idx):
        '''
        INPUT:
        src_idx (B, S); <bos> x <eos>
        trg_idx (B, T); <bos> y <eos>

        RETURN: 
        reconstruct_src (B, S-1, V) x_ <eos>
        reconstruct_trg (B, T-1, V) y_ <eos>
        '''

        ###############################################
        # first src to trg
        ###############################################
        
        # sequence to sequence learning

        src_pm, trg_pm = self._get_padding_mask(src_idx, trg_idx[:, :-1])
        _, trg_m, trg_src_m = self._get_causal_mask(src_idx, trg_idx[:, :-1])
            
        enc_src = self.encoder.encode(src_idx, None, src_pm)
        dec_out_trg = self.decoder.decode(trg_idx[:, :-1], enc_src, trg_m, trg_src_m, trg_pm, src_pm)

        # reconstruction src

        _, T = trg_idx.size()
        trg_idx_ = src_idx[:, 0].unsqueeze(1).detach() # B, 1

        for _ in range(T-1):
            _, trg_m, trg_src_m = self._get_causal_mask(src_idx, trg_idx_)
            _, trg_pm = self._get_padding_mask(src_idx, trg_idx_)
            trg_out = self.decoder.decode(trg_idx_, enc_src, trg_m, trg_src_m, trg_pm, src_pm)
            trg_new_idx = torch.argmax(trg_out[:, -1], dim=1).unsqueeze(1)
            trg_idx_ = torch.cat((trg_idx_, trg_new_idx), dim=1)
        
        final_trg = trg_idx_.detach()
        
        src_pm_n, trg_pm_n = self._get_padding_mask(final_trg, src_idx[:, :-1])
        _, trg_m_n, trg_src_m_n = self._get_causal_mask(final_trg, src_idx[:, :-1])
            
        enc_src_n = self.encoder.encode(final_trg, None, src_pm_n)
        reconstruct_src = self.decoder.decode(src_idx[:, :-1], enc_src_n, trg_m_n, trg_src_m_n, trg_pm_n, src_pm_n)

        ###############################################
        # then trg to src
        ###############################################

        src_pm_n, trg_pm_n = self._get_padding_mask(trg_idx, src_idx[:, :-1])
        _, trg_m_n, trg_src_m_n = self._get_causal_mask(trg_idx, src_idx[:, :-1])
            
        enc_src_n = self.encoder.encode(trg_idx, None, src_pm_n)
        dec_out_src = self.decoder.decode(src_idx[:, :-1], enc_src_n, trg_m_n, trg_src_m_n, trg_pm_n, src_pm_n)

        _, S = src_idx.size()
        src_idx_ = src_idx[:, 0].unsqueeze(1).detach() # B, 1

        for _ in range(S-1):
            _, trg_m, trg_src_m = self._get_causal_mask(trg_idx_, src_idx_)
            _, trg_pm = self._get_padding_mask(trg_idx_, src_idx_)
            src_out = self.decoder.decode(src_idx_, enc_src_n, trg_m, trg_src_m, trg_pm, src_pm_n)
            src_new_idx = torch.argmax(src_out[:, -1], dim=1).unsqueeze(1)
            src_idx_ = torch.cat((src_idx_, src_new_idx), dim=1)
        
        final_src = src_idx_.detach()

        src_pm_n, trg_pm_n = self._get_padding_mask(final_src, trg_idx[:, :-1])
        _, trg_m_n, trg_src_m_n = self._get_causal_mask(final_src, trg_idx[:, :-1])
            
        enc_src_n = self.encoder.encode(final_src, None, src_pm_n)
        reconstruct_trg = self.decoder.decode(trg_idx[:, :-1], enc_src_n, trg_m_n, trg_src_m_n, trg_pm_n, src_pm_n)

        return dec_out_src, dec_out_trg, reconstruct_src, reconstruct_trg

    def sequence_to_sequence(self, src_idx, trg_idx):
        '''
        INPUT:
        src (B, S); <bos> x <eos>
        trg (B, T); <bos> y <eos>

        Return: (B, T-1, V); x <eos>
        Return: (B, S-1, V); y <eos>
        ''' 

        # just src to trg
        src_pm, trg_pm = self._get_padding_mask(src_idx, trg_idx[:, :-1])
        _, trg_m, trg_src_m = self._get_causal_mask(src_idx, trg_idx[:, :-1])
            
        enc_src = self.encoder.encode(src_idx, None, src_pm)
        dec_out_trg = self.decoder.decode(trg_idx[:, :-1], enc_src, trg_m, trg_src_m, trg_pm, src_pm)
        
        return dec_out_trg
    
    def language_modelling(self, src_idx):
        '''
        INPUT:
        src (B, S); <bos> x <eos>

        Return: (B, T-1, V); x <eos>
        ''' 

        # just src to trg
        src_pm, _ = self._get_padding_mask(src_idx[:, :-1], src_idx[:, :-1])
        src_m, _, _ = self._get_causal_mask(src_idx[:, :-1], src_idx[:, :-1])
            
        out = self.prior.encode(src_idx[:, :-1], src_m, src_pm)
        
        return out

    def inference(self, src, max_len=50, deterministic=True):
        '''
        INPUT:
        src (B, S) <bos> x <eos>

        RETURN: 
        trg_ (B, max_len) <bos> y_ <eos>
        '''
        
        src_pm, _ = self._get_padding_mask(src, src)
        enc_src = self.encoder.encode(src, None, src_pm)
        trg_idx = src[:, 0].unsqueeze(1).detach() # B, 1

        for i in range(max_len-1):
            
            _, trg_m, trg_src_m = self._get_causal_mask(src, trg_idx)
            _, trg_pm = self._get_padding_mask(src, trg_idx)

            dec_out = self.decoder.decode(trg_idx, enc_src, trg_m, trg_src_m, trg_pm, src_pm) # B, t, V
            
            if deterministic:
                trg_idx_new = torch.argmax(dec_out, dim=2)[:, -1].unsqueeze(1)
            else:
                trg_idx_new = torch.argmax(F.gumbel_softmax(dec_out, tau=0.01, hard=True), dim=2)[:, -1].unsqueeze(1) 
            
            trg_idx = torch.cat((trg_idx, trg_idx_new), dim=1)
        
        return trg_idx
    
    def _reconstruction_loss(self, src, trg):
        '''
        Calculate reconstruction loss
        src: (B, T, V)  xxxx <eos>
        trg: (B, T) xxxx <eos>
        return loss in (B*T)
        '''

        _, _, V = src.size()
        return self.loss(src.contiguous().view(-1, V), trg.contiguous().view(-1))
    
    def _KL_loss(self, q, p):
        '''
        Calculate kl divergence loss between q and p 
        KL(q|p) = -E_{q}[\log(p)-\log(q)]
        q: (B, T, V) xxxx <eos> 
        p: (B, T, V)  xxxx <eos>

        Returns: loss in (B*T)
        '''
        q_ = torch.argmax(q, dim=2)
        return self._reconstruction_loss(p, q_) - self._reconstruction_loss(q, q_)
        # return self.nll_loss(torch.log(p), q) - self.nll_loss(torch.log(q), q)
