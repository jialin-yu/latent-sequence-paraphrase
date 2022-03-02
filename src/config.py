
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

        self.quora_max_len = 25
        self.mscoco_max_len = 20

        self.quora_min_freq = 25
        self.mscoco_min_freq = 20

        self.max_len = 100

        self.batch_size = 32
        self.vocab_size = None

        self.inf = float('inf')

        self.lm_id = None

        for name, value in kwargs.items():
            setattr(self, name, value)

        if self.lm_dir:
            lm_name = f'seed_{self.seed}_hard_{self.hard}_batchloss_{self.batch_loss}_unsupervised_{self.unsupervised}_pseudo_{self.use_pseudo}'
            self.lm_id = lm_name

