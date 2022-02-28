import torch.nn as nn
import torch

class Encoder(nn.Module):
    def __init__(self, configs):
        super().__init__()
        
        self.device = 'cuda' if configs.cuda == True else 'cpu'
        self.tok_emb = nn.Embedding(configs.vocab_size, configs.hid_dim, padding_idx=configs.pad_id)
        self.pos_emb = nn.Embedding(configs.max_len, configs.hid_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=configs.hid_dim, nhead=configs.n_heads, dropout=configs.dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=configs.n_lays)
        self.dropout = nn.Dropout(configs.dropout)
        self.register_buffer('scale', torch.sqrt(torch.FloatTensor([configs.hid_dim])))

    def soft_encode(self, src_prob):
        '''
        INPUT: 
        src_prob (B, S, V) in normalised form
        
        RETURN: 
        output (B, S, E)
        '''
        batch_size, seq_len, _ = src_prob.size()
        pos = torch.arange(0, seq_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        src = self.dropout((src_prob @ self.tok_emb.weight * self.scale) + self.pos_emb(pos))
        output = self.encoder(src)  
        
        return output

    def hard_encode(self, src, src_mask=None, src_key_padding_mask=None):
        '''
        INPUT: 
        src (B, S)
        src_msk: (S, S)
        src_key_padding_mask: (B, S)
        
        RETURN: 
        output (B, S, E)
        '''
        batch_size, src_len = src.size()
        # pos: batch, seq_len
        pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        # src: batch, seq_len, hid_dim
        src = self.dropout((self.tok_emb(src) * self.scale) + self.pos_emb(pos))
        output = self.encoder(src, src_mask, src_key_padding_mask)  
        return output