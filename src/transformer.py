import torch
import torch.nn as nn
import torch.nn.functional as F

from encoder import TransformerEncoder
from decoder import TransformerDecoder
from pos_emb import PosEmbedding
from tok_emb import TokEmbedding

from lm import LanguageModel

from sample import straight_through_softmax, gumbel_softmax
    

class Transformer(nn.Module):

    def __init__(self, configs):
        super(Transformer, self).__init__()
        
        self.device = configs.device
        self.pad_id = configs.pad_id
        self.vocab_size = configs.vocab_size
        
        self.prior = LanguageModel(configs)

        self.src_encoder = TransformerEncoder(configs)
        self.trg_decoder = TransformerDecoder(configs)
        
        self.trg_encoder = TransformerEncoder(configs)
        self.src_decoder = TransformerDecoder(configs)
        
        self.loss = nn.CrossEntropyLoss(ignore_index=self.pad_id, reduction="none")

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

    def decode_lm(self, model, src):
        '''
        INPUT:
        src (B, S-1); <bos> x
        src_emb (B, S-1, H); <bos> x
        RETURN:
        output (B, S-1, V) x_ <eos>
        '''
        src_m, _, _ = self._get_causal_mask(src, src)
        src_pm, _ = self._get_padding_mask(src, src)
        output = model.decode(src, src_m, src_pm)
        return output

    def encode_sample_decode(self, encoder, decoder, src, gumbel_max=False, temperature=None):
        '''
        Encode and decode for latent variable 

        With gumbel softmax sampling
        https://arxiv.org/pdf/1611.01144.pdf
        https://arxiv.org/pdf/1611.00712.pdf
        
        And straight through (hard/soft)
        https://arxiv.org/pdf/1308.3432.pdf

        if gumbel_max, temperature is required
        
        INPUT:
        src (B, S); <bos> x <eos>

        RETURN: 
        trg_ (B, T, V) <bos> y_ <eos>
        trg_logits (B, T-1, V) y_ <eos>
        '''
        _, T = src.size()
        
        src_pm, _ = self._get_padding_mask(src, src)
        enc_src = encoder.encode(src, None, src_pm)
        dec_temp= src.detach()[:, 0].unsqueeze(1) # B, 1

        for i in range(T-1):
            
            _, trg_m, trg_src_m = self._get_causal_mask(src, dec_temp)
            _, trg_pm = self._get_padding_mask(src, dec_temp)

            dec_out = decoder.decode(dec_temp, enc_src, trg_m, trg_src_m, trg_pm, src_pm)
                
            if gumbel_max:
                trg_new = torch.argmax(gumbel_softmax(dec_out, temperature), dim=2)[:, -1].unsqueeze(1)
                dec_temp = torch.cat((dec_temp, trg_new), dim=1)
                # dec_temp = torch.cat((dec_temp, trg_new), dim=1).detach()
            else:
                trg_new = torch.argmax(straight_through_softmax(dec_out), dim=2)[:, -1].unsqueeze(1)
                dec_temp = torch.cat((dec_temp, trg_new), dim=1)
                # dec_temp = torch.cat((dec_temp, trg_new), dim=1).detach()
            
        assert dec_temp.size() == src.size()
        return dec_temp, dec_out
    
    def encode_and_decode(self, encoder, decoder, src, trg):
        '''
        INPUT:
        src (B, S) i; <bos> x <eos>
        trg (B, T-1) <bos> y

        Return: (B, T-1, V) y_ <eos>
        '''
        if len(src.size()) == 2:
            src_pm, trg_pm = self._get_padding_mask(src, trg)
            _, trg_m, trg_src_m = self._get_causal_mask(src, trg)
            enc_src = encoder.encode(src, None, src_pm)
            dec_out = decoder.decode(trg, enc_src, trg_m, trg_src_m, trg_pm, src_pm)
        else:
            src_emb = src
            src = torch.argmax(src, dim=2)
            src_pm, trg_pm = self._get_padding_mask(src, trg)
            _, trg_m, trg_src_m = self._get_causal_mask(src, trg)
            enc_src = encoder.encode(src_emb, None, src_pm)
            dec_out = decoder.decode(trg, enc_src, trg_m, trg_src_m, trg_pm, src_pm)
        
        return dec_out

    def inference(self, encoder, decoder, src, max_len=150):
        '''
        INPUT:
        src (B, S) <bos> x <eos>

        RETURN: 
        trg_ (B, max_len) <bos> y_ <eos>
        '''
        
        src_pm, _ = self._get_padding_mask(src, src)
        enc_src = encoder.encode(src, None, src_pm)
        dec_temp= src[:, 0].unsqueeze(1) # B, 1

        for i in range(max_len-1):

            _, trg_m, trg_src_m = self._get_causal_mask(src, dec_temp)
            _, trg_pm = self._get_padding_mask(src, dec_temp)

            dec_out = decoder.decode(dec_temp, enc_src, trg_m, trg_src_m, trg_pm, src_pm)
            dec_out_ = torch.argmax(dec_out, dim=2)[:, -1].unsqueeze(1)
            dec_temp = torch.cat((dec_temp, dec_out_), dim=1)
        
        return dec_temp
    
    def _reconstruction_loss(self, src, trg):
        '''
        Calculate reconstruction loss
        src: (B, T, V)  xxxx <eos>
        trg: (B, T) xxxx <eos>
        return loss in (B*T)
        '''

        B, _, V = src.size()
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
