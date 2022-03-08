import torch.nn as nn
import torch.nn.functional as F
import torch

class Decoder(nn.Module):
    def __init__(self, configs):
        super().__init__()
        
        self.device = configs.device
        self.tok_emb = nn.Embedding(configs.vocab_size, configs.hid_dim)
        self.pos_emb = nn.Embedding(configs.max_len, configs.hid_dim)
        decoder_layer = nn.TransformerDecoderLayer(d_model=configs.hid_dim, nhead=configs.n_heads, dropout=configs.dropout, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=configs.n_lays)
        self.dropout = nn.Dropout(configs.dropout)
        self.register_buffer('scale', torch.sqrt(torch.FloatTensor([configs.hid_dim])))
        self.linear = nn.Linear(configs.hid_dim, configs.vocab_size)

    def decode(self, trg, src_memory, trg_m=None, trg_src_m=None, trg_pm=None, src_memory_pm=None, trg_hard=False):
        '''
        INPUT: 
        
        trg (B, T-1, V) if hard == False; <bos> y 
        trg (B, T-1) if hard == True; <bos> y
        src_memory (B, S, H) <bos> x <eos>
        trg_m (T-1, T-1)
        trg_src_m (T-1, S)
        trg_pm (B, T-1)
        src_memory_pm (B, S)
        
        RETURN: 

        trg_ (B, T-1); y_ <eos> 
        '''

        if trg_hard:
            B, T = trg.size()
            pos = torch.arange(0, T).unsqueeze(0).repeat(B, 1).to(self.device) 
            trg = self.dropout((self.tok_emb(trg) * self.scale) + self.pos_emb(pos))
            y_ = self.decoder(trg, src_memory, trg_m, trg_src_m, trg_pm, src_memory_pm) 
            # y_ = F.softmax(self.linear(y_), dim=-1)
            y_ = self.linear(y_)
        else:
            B, T, _ = trg.size()
            pos = torch.arange(0, T).unsqueeze(0).repeat(B, 1).to(self.device) 
            trg = self.dropout(((trg.double() @ self.tok_emb.weight.double())* self.scale) + self.pos_emb(pos))
            y_ = self.decoder(trg.float(), src_memory, trg_m, trg_src_m, trg_pm, src_memory_pm)
            # print('This line okayy')
            y_ = F.softmax(self.linear(y_), dim=-1)
        
        return y_