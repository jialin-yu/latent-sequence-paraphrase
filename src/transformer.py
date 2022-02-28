import imp
import torch
import torch.nn as nn
import torch.nn.functional as F

from encoder import Encoder
from decoder import Decoder
from lm import LanguageModel

from sample import straight_through_softmax, gumbel_softmax, gumbel_softmax_sample
    

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

    def _get_mask(self, src, trg):
        '''
        src: (B, S)
        trg: (B, T)
        for square mask, the masked one is float('-inf') and unmasked one is 0
        for key_pad_mask, the masked one is 0 and unmasked one is 1 [0000111]
        '''
        _, S = src.size()
        _, T = trg.size()
        
        src_msk = torch.zeros((S, S)).type(torch.bool).to(self.device)
        trg_msk = torch.triu(torch.ones(T, T) * float('-inf'), diagonal=1).to(self.device)
        memory_msk = torch.triu(torch.ones(T, S) * float('-inf'), diagonal=1).to(self.device) 
        src_key_padding_mask = (src == self.pad_id).to(self.device)
        trg_key_padding_mask = (trg == self.pad_id).to(self.device)
        
        return src_msk, trg_msk, memory_msk, src_key_padding_mask, trg_key_padding_mask 

    def encode_sample_decode(self, encoder, decoder, src, trg, hard=False, gumbel_max=False, temperature=None):
        '''
        Encode and decode with gumbel softmax sampling (hard/soft)
        https://arxiv.org/pdf/1611.01144.pdf
        https://arxiv.org/pdf/1611.00712.pdf
        And straight through (hard/soft)
        https://arxiv.org/pdf/1308.3432.pdf
        hard return in categorical variable format, soft return in categorical parameter format 
        if gumbel_max, temperature is required
        INPUT:
        src (B, S)
        trg (B, T)

        RETURN: 
        (B, T, V)
        '''
        B, T = trg.size()
        src_m, _, _, src_kp_m, _ = self._get_mask(src, trg)
        enc_src = self.src_encoder(src, src_m, src_kp_m)
        shape = enc_src.size()
        ind = torch.LongTensor([self.bos_id]).expand(shape[0], shape[1]).unsqueeze(1) # B, S, 1
        dec_head = torch.zeros_like(enc_src).view(-1, shape[-1])
        dec_head.scatter_(1, ind.view(-1, 1), 1)
        dec_head = dec_head.view(*shape)[:, 0, :].unsqueeze(1) # B,1, V
        dec_temp = dec_head

        for index in range(T-1):
            if hard:
                if gumbel_max:
                    dec_out = decoder.soft_decode(dec_temp, enc_src)
                    trg_new = gumbel_softmax(dec_out, temperature)[:, -1, :].unsqueeze(1)
                    dec_temp = torch.cat((dec_temp, trg_new), dim=1)
                else:
                    dec_out = decoder.soft_decode(dec_temp, enc_src)
                    trg_new = straight_through_softmax(dec_out)[:, -1, :].unsqueeze(1) 
                    dec_temp = torch.cat((dec_temp, trg_new), dim=1)
            else:
                if gumbel_max:
                    dec_out = decoder.soft_decode(dec_temp, enc_src)
                    trg_new = gumbel_softmax_sample(dec_out, temperature)[:, -1, :].unsqueeze(1) 
                    dec_temp = torch.cat((dec_temp, trg_new), dim=1)
                else:
                    dec_out = decoder.soft_decode(dec_temp, enc_src)
                    trg_new = F.softmax(dec_out, dim=-1)[:, -1, :].unsqueeze(1) # B, 1, V
                    dec_temp = torch.cat((dec_temp, trg_new), dim=1)
        
        # append the dec_head for (B, T, V)
        out = torch.cat((dec_temp, dec_out), dim=1)
        return out
    
    def encode_and_decode(self, encoder, decoder, src, trg, encode_hard=True):
        '''
        INPUT:
        src (B, S) if encode_hard == Ture
        src (B, S, V) if not encode_hard == False, must be normalised
        trg: (B, T)

        Return: (B, T-1, V) in normalised form
        '''
        if encode_hard:
            src_m, trg_m, memory_m, src_kp_m, trg_kp_m = self._get_mask(src, trg[:, :-1])
            enc_src = encoder.hard_encode(src, src_m, src_kp_m)
            dec_out = decoder.hard_decode(trg[:, :-1], enc_src, trg_m, memory_m, trg_kp_m, src_kp_m)
        else:
            enc_src = encoder.soft_encode(src)
            dec_out = decoder.hard_decode(trg[:, :-1], enc_src)
        
        return dec_out


    def _reconstruct_error(self, src, trg, hard=False, trg_mask=None):
        '''
        Calculate cross entropy based on src and trg
        src: (B, T, V)  xxxx <eos>
        trg: (B, T) xxxx <eos>
        trg_mask: (B, T) xxxx <eos>

        Returns: loss in (batch_size)
        '''

        if hard:
            _, S, V = src.size()
            log_p = F.log_softmax(src, dim=2)
            loss = -((log_p * F.one_hot(trg, V)).sum(2) * (1 - trg_mask.int())).sum(1)
            
            loss_ = self.loss(src.contiguous().view(-1, V), trg.contiguous().view(-1)) # B*S
            loss_ = loss_.contiguous().view(-1, S).sum(1)

            print(torch.mean(loss))
            print(torch.mean(loss_))
            
            # loss = - torch.mean((log_p * trg.unsqueeze(2)) * (1 - trg_mask.int())), dim=1)
            

            # log_p = F.log_softmax(src.contiguous().view(-1, V), dim=1)
            # unmask_loss = -(log_p * trg.contiguous().view(-1).unsqueeze(1)).sum(1).contiguous().view(B, -1)
            # loss = (unmask_loss * (1. - trg_mask.int())).sum(1)
            # loss = -((log_p * trg.contiguous().view(-1).unsqueeze(1)) * (1. - trg_mask.int().contiguous().view(-1))).contiguous().view(-1, S).sum(dim=1) 
        else: 
            _, S, V = src.size()
            loss = self.loss(src.contiguous().view(-1, V), trg.contiguous().view(-1)) # B*S
            loss = loss.contiguous().view(-1, S).sum(1)

        return loss

    def _log_probability(self, src, trg, hard=False, trg_mask=None):
        '''
        calculate soft/hard log probability
        return : (batch_size)
        '''
        return -self._reconstruct_error(src, trg, hard=False, trg_mask=None)