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

    def soft_lm(self, x_prob):
        '''
        INPUT: 
        x_prob (B, S, V) in normalised form
        
        RETURN: 
        output (B, S, V) in un-normalised form
        '''
        batch_size, seq_len, _ = x_prob.size()
        pos = torch.arange(0, seq_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        x = self.dropout((x_prob @ self.tok_emb.weight * self.scale) + self.pos_emb(pos))
        output = self.encoder(x)
        output = F.softmax(self.linear(output), dim=-1)
        
        return output

    def hard_lm(self, x, x_mask=None, x_key_padding_mask=None):
        '''
        INPUT: 
        x (B, S)
        x_msk: (S, S)
        x_key_padding_mask: (B, S)
        
        RETURN: 
        output (B, S, V) in normalised form
        '''
        batch_size, seq_len = x.size()
        # pos: batch, seq_len
        pos = torch.arange(0, seq_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        # src: batch, seq_len, hid_dim
        x = self.dropout((self.tok_emb(x) * self.scale) + self.pos_emb(pos))
        output = self.encoder(x, x_mask, x_key_padding_mask)  
        output = F.softmax(self.linear(output), dim=-1)

        return output