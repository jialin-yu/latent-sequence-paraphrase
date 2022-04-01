import torch.nn as nn
import torch
from pos_emb import PosEmbedding
from tok_emb import TokEmbedding

class Encoder(nn.Module):
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
            