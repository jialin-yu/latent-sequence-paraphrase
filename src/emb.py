import torch.nn as nn
import torch

class TokEmbedding(nn.Module):
    def __init__(self, configs, use_scale=True):
        super().__init__()
        
        self.device = configs.device
        self.use_scale = use_scale
        self.emb = nn.Embedding(configs.vocab_size, configs.hid_dim)
        self.register_buffer('scale', torch.sqrt(torch.FloatTensor([configs.hid_dim])))
    
    def forward(self, x):
        '''
        INPUT: x (B, S); or x (B, S, V)
        RETURN: x_ (B, S, H)  
        '''
        if len(x.size()) == 2:
            if self.use_scale:
                return self.emb(x) * self.scale
            else:
                return self.emb(x)
        else:
            assert len(x.size()) == 3
            if self.use_scale:
                return ((x.double() @ self.emb.weight.double()) * self.scale).float()
            else:
                return ((x.double() @ self.emb.weight.double())).float()

class PosEmbedding(nn.Module):
    def __init__(self, configs):
        super().__init__()
        
        self.device = configs.device        
        self.dropout = nn.Dropout(configs.dropout)
        self.pos_emb = nn.Embedding(configs.max_len, configs.hid_dim)
    
    def forward(self, x_tok):
        '''
        INPUT: x (B, S, H)
        RETURN: x_ (B, S, H)
        '''

        batch_size, trg_len, _ = x_tok.size()
        pos = torch.arange(0, trg_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        return self.dropout(x_tok + self.pos_emb(pos))

class Embedding(nn.Module):
    def __init__(self, configs, use_scale=True):
        super().__init__()

        self.tok_emb = TokEmbedding(configs, use_scale)
        self.pos_emb = PosEmbedding(configs)

    def forward(self, x):
        '''
        INPUT: x (B, S); or x (B, S, V)
        OUTPUT: x_ (B, S, H)
        '''
        return self.pos_emb(self.tok_emb(x))