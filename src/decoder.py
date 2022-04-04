import torch.nn as nn
import torch.nn.functional as F
import torch
from pos_emb import PosEmbedding
from tok_emb import TokEmbedding

class TransformerDecoder(nn.Module):
    def __init__(self, configs):
        super().__init__()
        
        self.device = configs.device
        self.pos_emb = PosEmbedding(configs)
        self.tok_emb = TokEmbedding(configs)
        decoder_layer = nn.TransformerDecoderLayer(d_model=configs.hid_dim, nhead=configs.n_heads, dropout=configs.dropout, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=configs.n_lays)
        self.dropout = nn.Dropout(configs.dropout)
        self.register_buffer('scale', torch.sqrt(torch.FloatTensor([configs.hid_dim])))
        self.linear = nn.Linear(configs.hid_dim, configs.vocab_size)

    def decode(self, trg, src_memory, trg_m=None, trg_src_m=None, trg_pm=None, src_memory_pm=None):
        '''
        INPUT: 
        
        trg (B, T-1); <bos> y 
        src_memory (B, S, H) <bos> x <eos>
        trg_m (T-1, T-1)
        trg_src_m (T-1, S)
        trg_pm (B, T-1)
        src_memory_pm (B, S)
        
        RETURN: 

        trg_ (B, T-1, V); y_ <eos> 
        '''

        return self.linear(self.decoder(self.pos_emb(self.tok_emb(trg)), src_memory, trg_m, trg_src_m, trg_pm, src_memory_pm))

class Seq2SeqDecoder(nn.Module):
    def __init__(self, configs):
        super().__init__()
        
        self.device = configs.device
        self.output_dim = configs.vocab_size
        self.hid_dim = configs.hid_dim
        self.n_layers = 2
        self.dropout = 0.3
        self.embedding = nn.Embedding(configs.vocab_size, configs.hid_dim)
        self.rnn = nn.LSTM(configs.hid_dim, configs.hid_dim, self.n_layers, dropout = self.dropout)
        self.fc_out = nn.Linear(configs.hid_dim, self.output_dim)
        self.dropout = nn.Dropout(self.dropout)

    def decode(self, input, hidden, cell):
        '''
        INPUT: 
        
        #input = [batch size]
        x_m (S, S) == None 
        x_pm (B, S)

        RETURN: 
        x_ (B, S, H); <bos> x_ <eos> 
        '''
        input = input.unsqueeze(0)
        embedded = self.dropout(self.embedding(input))
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        prediction = self.fc_out(output.squeeze(0))

        return prediction, hidden, cell
