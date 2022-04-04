import torch.nn as nn
import torch
from pos_emb import PosEmbedding
from tok_emb import TokEmbedding

class TransformerEncoder(nn.Module):
    def __init__(self, configs):
        super().__init__()
        
        self.device = configs.device
        self.pos_emb = PosEmbedding(configs)
        self.tok_emb = TokEmbedding(configs)
        encoder_layer = nn.TransformerEncoderLayer(d_model=configs.hid_dim, nhead=configs.n_heads, dropout=configs.dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=configs.n_lays)
        self.dropout = nn.Dropout(configs.dropout)
        self.register_buffer('scale', torch.sqrt(torch.FloatTensor([configs.hid_dim])))

    def encode(self, x, x_m=None, x_pm=None):
        '''
        INPUT: 
        
        x (B, S); <bos> x <eos>
        x_m (S, S) == None 
        x_pm (B, S)

        RETURN: 
        x_ (B, S, H); <bos> x_ <eos> 
        '''
        return self.encoder(self.pos_emb(self.tok_emb(x)), x_m, x_pm) 

class Seq2SeqEncoder(nn.Module):
    def __init__(self, configs):
        super().__init__()
        
        self.device = configs.device
        self.hid_dim = configs.hid_dim
        self.n_layers = 2
        self.dropout = 0.3
        self.embedding = nn.Embedding(configs.vocab_size, configs.hid_dim)
        self.rnn = nn.LSTM(configs.hid_dim, configs.hid_dim, self.n_layers, dropout = self.dropout)
        self.dropout = nn.Dropout(self.dropout)

    def encode(self, x):
        '''
        INPUT: 
        
        x (B, S); <bos> x <eos>
        x_m (S, S) == None 
        x_pm (B, S)

        RETURN: 
        x_ (B, S, H); <bos> x_ <eos> 
        '''
        embedded = self.dropout(self.embedding(x))
        outputs, (hidden, cell) = self.rnn(embedded)
        return hidden, cell