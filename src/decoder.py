import torch.nn as nn
import torch.nn.functional as F
import torch

class Decoder(nn.Module):
    def __init__(self, configs):
        super().__init__()
        
        self.device = 'cuda' if configs.cuda == True else 'cpu'
        self.tok_emb = nn.Embedding(configs.vocab_size, configs.hid_dim, padding_idx=configs.pad_id)
        self.pos_emb = nn.Embedding(configs.max_len, configs.hid_dim)
        decoder_layer = nn.TransformerDecoderLayer(d_model=configs.hid_dim, nhead=configs.n_heads, dropout=configs.dropout, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=configs.n_lays)
        self.dropout = nn.Dropout(configs.dropout)
        self.register_buffer('scale', torch.sqrt(torch.FloatTensor([configs.hid_dim])))
        self.linear = nn.Linear(configs.hid_dim, configs.vocab_size)

    def soft_decode(self, tgt_distr, memory):
        '''
        INPUT:
        tgt_distr (B, T, V) in normalised form
        memory (B, S, E)

        RETURN:
        output (B, T, V) in in normalised form
        '''
        # single step decode for tgt probability
        batch_size, seq_len, _ = tgt_distr.size()
        pos = torch.arange(0, seq_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        tgt = self.dropout((tgt_distr @ self.tok_emb.weight * self.scale) + self.pos_emb(pos))
        output = self.decoder(tgt, memory) 
        output = F.softmax(self.linear(output), dim=-1)
        return output
    
    def hard_decode(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        '''
        INPUT:
        tgt (B, T)
        memory (B, S, E) 
        tgt_mask (T, T)
        memory_mask (T, S)
        tgt_key_padding_mask (B, T)
        memory_key_padding_mask (B, S)

        RETURN:
        output (B, T, V) in in normalised form
        '''

        batch_size, trg_len = tgt.size()
        # pos: batch, seq_len
        pos = torch.arange(0, trg_len).unsqueeze(0).repeat(batch_size, 1).to(tgt.device)
        # src: batch, seq_len, hid_dim
        tgt = self.dropout((self.tok_emb(tgt) * self.scale) + self.pos_emb(pos))
        output = self.decoder(tgt, memory, tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask) 
        output = F.softmax(self.linear(output), dim=-1)
        
        return output