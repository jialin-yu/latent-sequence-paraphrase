import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

class LSTM_LM(nn.Module):

    def __init__(self, configs):
        super(LSTM_LM, self).__init__()
        self.d_hidden = configs.lm_d_hidden
        self.embed = nn.Embedding(configs.vocab_size, configs.lm_d_word_vec , padding_idx=configs.pad_id)
        self.dropout_in = nn.Dropout(configs.dropout_in)
        self.dropout_out = nn.Dropout(configs.dropout_out)

        self.lstm = nn.LSTM(input_size=configs.lm_d_word_vec, hidden_size=configs.lm_d_hidden, num_layers=1, batch_first=True)
        self.pred_linear = nn.Linear(configs.lm_d_hidden, configs.vocab_size, bias=True)

        self.loss = nn.CrossEntropyLoss(ignore_index=configs.pad_id, reduction="none")

    def decode(self, x, x_len, gumbel_softmax=False):
        '''
        Core decode function
        Accept two types of input for x:
        (1) single index form x: (batch_size, seq_len)
        (2) gumbel_softmax form x: (batch_size, seq_len, vocab_size)
        x_len: list of x lengths

        Returns: (batch_size, seq_len, vocab_size)
        '''

        if gumbel_softmax:
            batch_size, seq_len, _ = x.size()
            word_embed = x @ self.embed.weight
        else:
            batch_size, seq_len = x.size()
            word_embed = self.embed(x)

        word_embed = self.dropout_in(word_embed)
        packed_embed = pack_padded_sequence(word_embed, x_len.to('cpu'), batch_first=True)
        
        c_init = word_embed.new_zeros((1, batch_size, self.d_hidden))
        h_init = word_embed.new_zeros((1, batch_size, self.d_hidden))
        output, _ = self.lstm(packed_embed, (h_init, c_init))
        output, _ = pad_packed_sequence(output, batch_first=True)
        output = self.dropout_out(output)
        output_logits = self.pred_linear(output)

        return output_logits

    def reconstruct_error(self, x, x_len, gumbel_softmax=False, x_mask=None):
        '''
        Calculate cross entropy
        Accept x, x_len similar as in decode function
        x_mask: required, if gumbel_softmax is True. 1 denotes mask, size: (batch_size, seq_len)

        Returns: loss in (batch_size)
        '''

        #remove end symbol
        src = x[:, :-1]
        # remove start symbol
        tgt = x[:, 1:]
        # sentence len reduce by 1
        x_len = [s - 1 for s in x_len]
        # decode reduced src
        output_logits = self.decode(src, x_len, gumbel_softmax)

        if gumbel_softmax:
            batch_size, seq_len, _ = src.size()
            log_p = F.log_softmax(output_logits, dim=2)
            x_mask = x_mask[:, 1:]
            loss = -((log_p * tgt).sum(dim=2) * (1. - x_mask)).sum(dim=1)
        else:
            batch_size, seq_len = src.size()
            tgt = tgt.contiguous().view(-1)
            loss = self.loss(output_logits.view(-1, output_logits.size(2)),tgt)
        
        return loss

    def log_probability(self, x, x_len, gumbel_softmax=False, x_mask=None):
        '''
        calculate soft/hard log probability
        return : (batch_size)
        '''
        return -self.reconstruct_error(x, x_len, gumbel_softmax, x_mask)