from numpy import argmax
import torch
import torch.nn as nn
import torch.nn.functional as F

from encoder import RNNEncoder
from decoder import RNNDecoder

from lm import LanguageModel

from sample import straight_through_softmax, gumbel_softmax_sample
    

class LSTM(nn.Module):

    def __init__(self, configs):
        super(LSTM, self).__init__()
        
        self.device = configs.device
        self.pad_id = configs.pad_id
        self.vocab_size = configs.vocab_size
        
        self.prior = LanguageModel(configs)

        self.src_encoder = RNNEncoder(configs)
        self.trg_decoder = RNNDecoder(configs)
        
        self.trg_encoder = RNNEncoder(configs)
        self.src_decoder = RNNDecoder(configs)
        
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
        B, T = src.size()
        
        # src_pm, _ = self._get_padding_mask(src, src)
        # enc_src = encoder.encode(src, None, src_pm)
        hidden, cell = encoder.encode(src)
        dec_temp_hard = src[:, 0].unsqueeze(1).detach() # B, 1
        dec_temp= F.one_hot(src, self.vocab_size)[:, 0].unsqueeze(1).double().detach() # B, 1, V
        outputs = torch.zeros(T, B, self.vocab_size).to(self.device)
        
        for t in range(T-1):
            
            # _, trg_m, trg_src_m = self._get_causal_mask(src, dec_temp_hard)
            # _, trg_pm = self._get_padding_mask(src, dec_temp_hard)

            output, hidden, cell = self.decoder(dec_temp, hidden, cell)

            # outputs[t] = output
            # input = src[t]

            # teacher_force = random.random() < teacher_forcing_ratio

            # dec_out = decoder.decode(dec_temp, enc_src, trg_m, trg_src_m, trg_pm, src_pm) # B, t+1, V
                
            if gumbel_max:
                # new_temp = gumbel_softmax_sample(dec_out, temperature)[:, -1].unsqueeze(1) # B, 1, V
                new_temp = gumbel_softmax_sample(output, temperature)[:, -1].unsqueeze(1).detach() # B, 1, V
                dec_temp = torch.cat((dec_temp, new_temp), dim=1).detach() # B, t+2, V
                new_temp_hard = torch.argmax(new_temp, dim=2)[:, -1].unsqueeze(1).detach()
                dec_temp_hard = torch.cat((dec_temp_hard, new_temp_hard), dim=1).detach()

                outputs[t] = new_temp_hard
                # trg_new = torch.argmax(gumbel_softmax(dec_out, temperature), dim=2)[:, -1].unsqueeze(1)
                # dec_temp = torch.cat((dec_temp, trg_new), dim=1)
                # dec_temp = torch.cat((dec_temp, trg_new), dim=1).detach()
            else:
                new_temp = straight_through_softmax(output)[:, -1].unsqueeze(1) # B, 1, V
                dec_temp = torch.cat((dec_temp, new_temp), dim=1).detach() # B, t+2, V
                new_temp_hard = torch.argmax(new_temp, dim=2)[:, -1].unsqueeze(1).detach()
                dec_temp_hard = torch.cat((dec_temp_hard, new_temp_hard), dim=1).detach()

                outputs[t] = new_temp_hard

                # trg_new = torch.argmax(straight_through_softmax(dec_out), dim=2)[:, -1].unsqueeze(1)
                # dec_temp = torch.cat((dec_temp, strg_new), dim=1)
                # dec_temp = torch.cat((dec_temp, trg_new), dim=1).detach()
        
        final_output = output # B, T-1, V

        assert dec_temp.size()[:-1] == src.size()
        assert dec_temp_hard.size() == src.size()
        return dec_temp, dec_temp_hard, final_output
    
    def encode_and_decode(self, encoder, decoder, src, trg):
        '''
        INPUT:
        src (B, S) i; <bos> x <eos>
        trg (B, T-1) <bos> y

        Return: (B, T-1, V) y_ <eos>
        '''

        if len(src.size()) == 2:
            # src_pm, trg_pm = self._get_padding_mask(src, trg)
            # _, trg_m, trg_src_m = self._get_causal_mask(src, trg)
            # enc_src = encoder.encode(src, None, src_pm)
            hidden, cell = encoder.encode(src)
            # output, hidden, cell = decoder.decode(src, hidden, cell)
        else:
            src_hard = torch.argmax(src, dim=2)
            # src_pm, trg_pm = self._get_padding_mask(src_hard, trg)
            # _, trg_m, trg_src_m = self._get_causal_mask(src_hard, trg)
            hidden, cell = encoder.encode(src_hard)
            # output, hidden, cell = decoder(src, hidden, cell)

        B, T = trg.size()
        outputs = torch.zeros(B, T, self.vocab_size).to(self.device)
        

        for t in range(T):

            input = trg[:, t]
            #insert input token embedding, previous hidden and previous cell states
            #receive output tensor (predictions) and new hidden and cell states
            output, hidden, cell = decoder.decode(input, hidden, cell)
            
            #place predictions in a tensor holding predictions for each token
            outputs[:, t] = output
            
            #if teacher forcing, use actual next token as next input
            #if not, use predicted token
        
        return outputs
        

    def inference(self, encoder, decoder, src, max_len=150):
        '''
        INPUT:
        src (B, S) <bos> x <eos>

        RETURN: 
        trg_ (B, max_len) <bos> y_ <eos>
        '''
        
        # src_pm, _ = self._get_padding_mask(src, src)
        # enc_src = encoder.encode(src, None, src_pm)
        hidden, cell = encoder.encode(src)
        
        B, _ = src.size()
        outputs = torch.zeros(B, max_len, self.vocab_size).to(self.device)

        input = src[:, 0]
        output, hidden, cell = decoder.decode(input, hidden, cell)
        outputs[:, 0] = output

        for i in range(1, max_len):

            # _, trg_m, trg_src_m = self._get_causal_mask(src, dec_temp)
            # _, trg_pm = self._get_padding_mask(src, dec_temp)
            input = torch.argmax(output, dim=1)
            output, hidden, cell = decoder.decode(input, hidden, cell)
            outputs[:, i] = output

        return  torch.argmax(outputs, dim=2)
    
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
