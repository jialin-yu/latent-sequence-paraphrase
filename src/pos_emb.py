import torch.nn as nn
import torch
import math

class PosEmbedding(nn.Module):
    def __init__(self, configs):
        super().__init__()
        
        self.device = configs.device

        #den = torch.exp(- torch.arange(0, configs.hid_dim, 2)* math.log(10000) / configs.hid_dim)
        #pos = torch.arange(0, configs.max_len).reshape(configs.max_len, 1)
        #pos_embedding = torch.zeros((configs.max_len, configs.hid_dim))
        #pos_embedding[:, 0::2] = torch.sin(pos * den)
        #pos_embedding[:, 1::2] = torch.cos(pos * den)
        #pos_embedding = pos_embedding.unsqueeze(0)
        #self.register_buffer('emb', pos_embedding)
        
        self.dropout = nn.Dropout(configs.dropout)
        self.pos_emb = nn.Embedding(configs.max_len, configs.hid_dim)
    
    def forward(self, x_tok):
        '''
        INPUT: 
        x (B, S, H) <bos> x <eos>
        RETURN: 
        x_ (B, S, H); <bos> x_ <eos> 
        '''
        # print(x_tok.size())
        # print(self.emb[:, :x_tok.size(1), :].size(
        
        batch_size, trg_len, _ = x_tok.size()
        pos = torch.arange(0, trg_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)

        return self.dropout(x_tok + self.pos_emb(pos))


