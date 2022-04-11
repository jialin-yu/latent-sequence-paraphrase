import torch.nn as nn
import torch

class TransformerEncoder(nn.Module):
    def __init__(self, configs):
        super().__init__()
        
        self.device = configs.device
        encoder_layer = nn.TransformerEncoderLayer(d_model=configs.hid_dim, nhead=configs.n_heads, dropout=configs.dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=configs.n_lays)

    def encode(self, x_emb, x_m=None, x_pm=None):
        '''
        INPUT: 
        x_emb (B, S, H); <bos> x <eos>
        x_m (S, S) == None 
        x_pm (B, S)

        RETURN: 
        x_ (B, S, H); <bos> x_ <eos> 
        '''
        return self.encoder(x_emb, x_m, x_pm) 

class RNNEncoder(nn.Module):
    def __init__(self, configs):
        super().__init__()
        
        self.device = configs.device
        self.encoder = nn.LSTM(configs.hid_dim, configs.hid_dim, configs.n_lays, dropout=configs.dropout, batch_first=True)
        self.dropout = nn.Dropout(configs.dropout)

    def encode(self, x_emb, x_len):
        '''
        INPUT: 
        x (B, S); <bos> x <eos>

        RETURN: 
        x_ (B, S, H); <bos> x_ <eos> 
        '''
        embedded = self.dropout(x_emb)
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, x_len.to('cpu'), batch_first=True, enforce_sorted=False)
        packed_outputs, (hidden, cell) = self.encoder(packed_embedded)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs, batch_first=True) 
        # outputs, (hidden, cell) = self.encoder(embedded)
        return hidden, cell