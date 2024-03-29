import torch.nn as nn
import torch

class TransformerEncoder(nn.Module):
    def __init__(self, configs):
        super().__init__()

        self.device = configs.device
        self.tok_emb = nn.Embedding(configs.vocab_size, configs.hid_dim)
        self.pos_emb = nn.Embedding(configs.max_len, configs.hid_dim)
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
        if len(x.size()) == 2:
            B, S = x.size()
            pos = torch.arange(0, S).unsqueeze(0).repeat(B, 1).to(self.device)
            x = self.dropout((self.tok_emb(x) * self.scale) + self.pos_emb(pos))
            x_ = self.encoder(x, x_m, x_pm)  
        else:
            assert len(x.size()) == 3
            B, S, _ = x.size()
            pos = torch.arange(0, S).unsqueeze(0).repeat(B, 1).to(self.device)
            x = self.dropout(((x.double() @ self.tok_emb.weight.double())* self.scale) + self.pos_emb(pos))
            x_ = self.encoder(x.float(), x_m, x_pm)  
            
        return x_