import torch.nn as nn
import torch

class TokEmbedding(nn.Module):
    def __init__(self, configs):
        super().__init__()
        
        self.device = configs.device
        self.emb = nn.Embedding(configs.vocab_size, configs.hid_dim)
        self.dropout = nn.Dropout(configs.dropout)
        self.register_buffer('scale', torch.sqrt(torch.FloatTensor([configs.hid_dim])))
    
    def forward(self, x):
        '''
        INPUT: 
        
        x (B, S, V) if hard == False; <bos> x <eos>
        x (B, S) if hard == True; <bos> x <eos>

        RETURN: 

        x_ (B, S, H); <bos> x_ <eos> 
        '''
        if len(x.size()) == 2:
            B, S = x.size()
            return self.emb(x) * self.scale
        else:
            assert len(x.size()) == 3
            B, S, _ = x.size()
            return (x.double() @ self.emb.weight.double())* self.scale
