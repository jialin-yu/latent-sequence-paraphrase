from distutils.command.config import config
import torch
import torch.nn as nn
import torch.nn.functional as F
from emb import Embedding

class LanguageModel(nn.Module):

    def __init__(self, configs):
        super(LanguageModel, self).__init__()

        self.device = configs.device

        # self.pos_emb = PosEmbedding(configs)
        # self.tok_emb = TokEmbedding(configs)
        encoder_layer = nn.TransformerEncoderLayer(d_model=configs.hid_dim, nhead=configs.n_heads, dropout=configs.dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=configs.n_lays)
        self.dropout = nn.Dropout(configs.dropout)
        self.register_buffer('scale', torch.sqrt(torch.FloatTensor([configs.hid_dim])))
        self.linear = nn.Linear(configs.hid_dim, configs.vocab_size)
    
    def decode(self, x, x_m=None, x_pm=None):
        '''
        INPUT: 
        x (B, S-1); <bos> x
        x_m (S-1, S-1)
        x_pm (B, S-1)

        RETURN: 
        x_ (B, S-1); x_ <eos> 
        '''
        return self.linear(self.encoder(self.pos_emb(self.tok_emb(x)), x_m, x_pm))