import imp
import torch
import torch.nn as nn
import torch.nn.functional as F

from encoder import Encoder
from decoder import Decoder
from lm import LanguageModel

from sample import straight_through_softmax, gumbel_softmax, gumbel_softmax_sample, straight_through_logits
    

class Transformer(nn.Module):

    def __init__(self, configs):
        super(Transformer, self).__init__()

        self.device = 'cuda' if configs.cuda==True else 'cpu'
        
        self.bos_id = configs.bos_id
        self.pad_id = configs.pad_id
        self.vocab_size = configs.vocab_size
        
        self.prior = LanguageModel(configs)

        self.src_encoder = Encoder(configs)
        self.trg_decoder = Decoder(configs)
        
        self.trg_encoder = Encoder(configs)
        self.src_decoder = Decoder(configs)
        
        # the loss should return in per batch
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

    def decode_lm(self, lm_model, src, src_hard=False):
        '''
        Decode src in lm_model

        INPUT:

        src (B, S-1) if hard == True; <bos> x
        src (B, S-1, V) if hard == False; <bos> x 


        RETURN:
        output (B, S-1, V) x_ <eos>
        '''
        if src_hard:
            src_m, _, _ = self._get_causal_mask(src, src)
            src_pm, _ = self._get_padding_mask(src, src)
            output = lm_model.decode(src, src_m, src_pm, True)
        else:
            hard_src = torch.argmax(src, dim=2)
            src_m, _, _ = self._get_causal_mask(hard_src, hard_src)
            src_pm, _ = self._get_padding_mask(hard_src, hard_src)
            output = lm_model.decode(src, src_m, src_pm, False)
        
        return output

    def encode_sample_decode(self, encoder, decoder, src, trg, latent_hard=False, gumbel_max=False, temperature=None):
        '''
        Encode and decode for latent variable 

        With gumbel softmax sampling
        https://arxiv.org/pdf/1611.01144.pdf
        https://arxiv.org/pdf/1611.00712.pdf
        
        And straight through (hard/soft)
        https://arxiv.org/pdf/1308.3432.pdf

        If latent_hard in 'one-hot' format; else in 'soft' format

        if gumbel_max, temperature is required
        
        INPUT:
        src (B, S) <bos> x <eos>
        trg (B, T) <bos> y <eos>

        RETURN: 
        trg_ (B, T, V) <bos> y_ <eos>
        trg_logits (B, T-1, V) y_ <eos>
        '''
        _, T = trg.size()
        
        src_pm, _ = self._get_padding_mask(src, src)
        enc_src = encoder.encode(src, None, src_pm, True)
        dec_temp= F.one_hot(src, self.vocab_size)[:, 0].unsqueeze(1).double().detach() # B, 1, V

        for i in range(T-1):

            hard_trg = torch.argmax(dec_temp, dim=2)
            _, trg_m, trg_src_m = self._get_causal_mask(src, hard_trg)
            _, trg_pm = self._get_padding_mask(src, hard_trg)

            if latent_hard:

                dec_out = decoder.decode(hard_trg, enc_src, trg_m, trg_src_m, trg_pm, src_pm, True)
                
                if gumbel_max:
                    trg_new = gumbel_softmax(dec_out, temperature)[:, -1].unsqueeze(1)
                    dec_temp = torch.cat((dec_temp, trg_new), dim=1).detach()
                else:
                    trg_new = straight_through_softmax(dec_out)[:, -1].unsqueeze(1) 
                    dec_temp = torch.cat((dec_temp, trg_new), dim=1).detach()
            else:

                dec_out = decoder.decode(dec_temp, enc_src, trg_m, trg_src_m, trg_pm, src_pm, False)
                
                if gumbel_max:
                    trg_new = gumbel_softmax_sample(dec_out, temperature)[:, -1].unsqueeze(1) 
                    dec_temp = torch.cat((dec_temp, trg_new), dim=1).detach()
                else:
                    trg_new = F.softmax(dec_out, dim=-1)[:, -1].unsqueeze(1)
                    dec_temp = torch.cat((dec_temp, trg_new), dim=1).detach()
        
        # dec_temp should have size (B, T, V)
        # print(dec_temp.size()[:-1])
        # print(trg.size())
        assert dec_temp.size()[:-1] == trg.size()
        # out = torch.cat((dec_temp, dec_out), dim=1)
        return dec_temp, dec_out
    
    def encode_and_decode(self, encoder, decoder, src, trg, src_hard=True):
        '''
        INPUT:
        src (B, S) if encode_hard == Ture; <bos> x <eos>
        src (B, S, V) if encode_hard == False; <bos> x <eos>
        trg (B, T-1) <bos> y

        Return: (B, T-1, V) y_ <eos>
        '''
        if src_hard:
            src_pm, trg_pm = self._get_padding_mask(src, trg)
            _, trg_m, trg_src_m = self._get_causal_mask(src, trg)
            enc_src = encoder.encode(src, None, src_pm, True)
            dec_out = decoder.decode(trg, enc_src, trg_m, trg_src_m, trg_pm, src_pm, True)
        else:
            src_hard_form = torch.argmax(src, dim=2)
            src_pm, trg_pm = self._get_padding_mask(src_hard_form, trg)
            _, trg_m, trg_src_m = self._get_causal_mask(src_hard_form, trg)
            enc_src = encoder.encode(src, None, src_pm, False)
            dec_out = decoder.decode(trg, enc_src, trg_m, trg_src_m, trg_pm, src_pm, True)
        
        return dec_out

    def inference(self, encoder, decoder, src, max_len=100):
        '''
        INPUT:
        src (B, S) <bos> x <eos>

        RETURN: 
        trg_ (B, max_len) <bos> y_ <eos>
        '''
        
        src_pm, _ = self._get_padding_mask(src, src)
        enc_src = encoder.encode(src, None, src_pm, True)
        dec_temp= src[:, 0].unsqueeze(1) # B, 1

        for i in range(max_len-1):

            _, trg_m, trg_src_m = self._get_causal_mask(src, dec_temp)
            _, trg_pm = self._get_padding_mask(src, dec_temp)

            dec_out = decoder.decode(dec_temp, enc_src, trg_m, trg_src_m, trg_pm, src_pm, True)
            dec_out_ = torch.argmax(dec_out, dim=2)[:, -1].unsqueeze(1)
            dec_temp = torch.cat((dec_temp, dec_out_), dim=1)
        
        return dec_temp
    
    def _reconstruction_loss(self, src, trg, hard_loss=False):
        '''
        Calculate reconstruction loss
        src: (B, T, V)  xxxx <eos>
        trg: (B, T) xxxx <eos>

        if hard_loss: src: (B, T, V) in one-hot format
        if penalty, return loss in (B)
        if not penalty, return loss in (B*T)
        '''

        B, _, V = src.size()

        if hard_loss:
            src = straight_through_logits(src)
        
        loss = self.loss(src.contiguous().view(-1, V), trg.contiguous().view(-1))
        
        return loss
    
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
    
    def _JSD_loss(self, q, p):
        '''
        Calculate Jensen–Shannon divergence loss between q and p 
        JSD(p|q) = 1/2*KL(P|M) + 1/2*KL(Q|M)
        q: (B, T, V) xxxx <eos> 
        p: (B, T, V)  xxxx <eos>

        Returns: loss in (B*T)
        '''
        m = 0.5*(q+p)
        return 0.5 * self._KL_loss(q, m) + 0.5 * self._KL_loss(p, m) 
