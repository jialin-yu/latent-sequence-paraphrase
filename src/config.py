import os

class Configs(object):
    def __init__(self, **kwargs):
        
        self.quora_fp = '../.data/quora_duplicate_questions.tsv'
        self.mscoco_fp_train = '../.data/annotations/captions_train2014.json'
        self.mscoco_fp_test = '../.data/annotations/captions_val2014.json'
        
        self.quora_max_len = 20
        self.quora_train_max = 115000
        self.quora_valid = 3000
        self.quora_test = 30000
        self.qu_batch_size = 512
        
        self.mscoco_max_len = 20
        self.mscoco_train_max = 115000
        self.mscoco_valid = 3000
        self.mscoco_test = 5000
        self.ms_batch_size = 512

        self.lm_dir = '../model/lm/'
        self.semi_dir = '../model/semi/'
        self.seq2seq_dir = '../model/seq2seq/'

        self.max_len = max(self.mscoco_max_len, self.quora_max_len) + 2

        self.vocab_size = None
        self.pad_id = None

        self.inf = float('inf')

        self.grad_clip = 1

        self.device = None

        self.fixed_temperature = None

        for dir in [self.lm_dir, self.semi_dir, self.seq2seq_dir]:
            if not os.path.exists(dir):
                os.makedirs(dir)

        for name, value in kwargs.items():
            setattr(self, name, value)
        
        lm_id = f'Data_{self.data}_Lr_{self.lm_lr}_Seed_{self.seed}_Ep_{self.lm_max_epoch}'
        self.lm_id = lm_id

        seq2seq_id = f'Data_{self.data}_Train_{self.train_size}_Lr_{self.seq2seq_lr}_Seed_{self.seed}_Duo_{self.duo}_Ep_{self.seq2seq_max_epoch}'
        self.seq2seq_id = seq2seq_id

        semi_id = f'Data_{self.data}_UNTrain_{self.un_train_size}_Train_{self.train_size}_Lr_{self.semi_lr}_Seed_{self.seed}_Gum_{self.gumbel_max}_Ep_{self.semi_max_epoch}'
        self.semi_id = semi_id
        
        

