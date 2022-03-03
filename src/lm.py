import torch
import torch.nn as nn
import torch.nn.functional as F

class LanguageModel(nn.Module):

    def __init__(self, configs):
        super(LanguageModel, self).__init__()

        self.device = 'cuda' if configs.cuda == True else 'cpu'
        self.tok_emb = nn.Embedding(configs.vocab_size, configs.hid_dim, padding_idx=configs.pad_id)
        self.pos_emb = nn.Embedding(configs.max_len, configs.hid_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=configs.hid_dim, nhead=configs.n_heads, dropout=configs.dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=configs.n_lays)
        self.dropout = nn.Dropout(configs.dropout)
        self.register_buffer('scale', torch.sqrt(torch.FloatTensor([configs.hid_dim])))
        self.linear = nn.Linear(configs.hid_dim, configs.vocab_size)
    
    def decode(self, x, x_m=None, x_pm=None, hard=False):
        '''
        INPUT: 
        
        x (B, S-1, V) if hard == False; <bos> x
        x (B, S-1) if hard == True; <bos> x
        x_m (S-1, S-1)
        x_pm (B, S-1)

        RETURN: 

        x_ (B, S-1); x_ <eos> 
        '''

        if hard:
            B, S = x.size()
            pos = torch.arange(0, S).unsqueeze(0).repeat(B, 1).to(self.device)
            x = self.dropout((self.tok_emb(x) * self.scale) + self.pos_emb(pos))
            x_ = self.encoder(x, x_m, x_pm)  
            x_ = F.softmax(self.linear(x_), dim=-1)
        else:
            B, S, _ = x.size()
            pos = torch.arange(0, S).unsqueeze(0).repeat(B, 1).to(self.device)
            x = self.dropout((x @ self.tok_emb.weight * self.scale) + self.pos_emb(pos))
            x_ = self.encoder(x, x_m, x_pm)  
            x_ = F.softmax(self.linear(x_), dim=-1)
        
        return x_