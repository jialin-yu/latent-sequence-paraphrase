import os

class Configs(object):
    def __init__(self, **kwargs):
        
        self.unk = '<unk>'
        self.pad = '<pad>'
        self.bos = '<bos>'
        self.eos = '<eos>'
        self.unk_id = 0
        self.pad_id = 1
        self.bos_id = 2
        self.eos_id = 3
        self.special_token = [self.unk, self.pad, self.bos, self.eos]

        self.quora_fp = '../.data/quora_duplicate_questions.tsv'
        self.mscoco_fp_train = '../.data/annotations/captions_train2014.json'
        self.mscoco_fp_test = '../.data/annotations/captions_val2014.json'

        self.max_vocab = None
        
        self.quora_max_len = 20
        self.quora_min_freq = 1
        self.quora_train_max = 50000
        self.quora_valid = 3000
        self.quora_test = 20000
        self.qu_batch_size = 16
        

        self.mscoco_max_len = 20
        self.mscoco_min_freq = 1
        self.mscoco_train_max = 75000
        self.mscoco_valid = 5000
        self.mscoco_test = 40000
        self.ms_batch_size = 32
       

        self.use_spacy = False

        self.max_len = max(self.mscoco_max_len, self.quora_max_len) + 2

        self.vocab_size = None

        self.inf = float('inf')

        self.grad_clip = 1

        self.device = None

        self.fixed_temperature = None

        for name, value in kwargs.items():
            setattr(self, name, value)

        if self.lm_dir:
            if not os.path.exists(self.lm_dir):
                os.makedirs(self.lm_dir)
            lm_id = f'D_{self.data}_L_{self.lm_lr}_S_{self.seed}_P_{self.use_pseudo}_EP_{self.lm_max_epoch}'
            self.lm_id = lm_id

        if self.vae_dir:
            if not os.path.exists(self.vae_dir):
                os.makedirs(self.vae_dir)
            vae_id = f'D_{self.data}_UNTR_{self.un_train_size}_L_{self.vae_lr}_S_{self.seed}_P_{self.use_pseudo}_HL_{self.latent_hard}_GM_{self.gumbel_max}_EP_{self.vae_max_epoch}'
            self.vae_id = vae_id

        if self.seq2seq_dir:
            if not os.path.exists(self.seq2seq_dir):
                os.makedirs(self.seq2seq_dir)
            seq2seq_id = f'D_{self.data}_TR_{self.train_size}_L_{self.seq2seq_lr}_S_{self.seed}_DUO_{self.duo}_EP_{self.seq2seq_max_epoch}'
            self.seq2seq_id = seq2seq_id

        if self.semi_dir:
            if not os.path.exists(self.semi_dir):
                os.makedirs(self.semi_dir)
            semi_id = f'D_{self.data}_UNTR_{self.un_train_size}_TR_{self.train_size}_L_{self.semi_lr}_S_{self.seed}_P_{self.use_pseudo}_HL_{self.latent_hard}_GM_{self.gumbel_max}_EP_{self.semi_max_epoch}'
            self.semi_id = semi_id
        
        

