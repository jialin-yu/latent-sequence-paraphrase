
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

        self.quora_min_freq = 5
        self.mscoco_min_freq = 5

        self.max_len = 150

        self.batch_size = 16
        self.vocab_size = None

        self.inf = float('inf')

        self.grad_clip = 1

        self.device = None

        for name, value in kwargs.items():
            setattr(self, name, value)

        if self.lm_dir:
            lm_name = f'seed_{self.seed}_hardloss_{self.hard_loss}_batchloss_{self.batch_loss}_unsupervised_{self.unsupervised}_pseudo_{self.use_pseudo}'
            self.lm_id = lm_name

        if self.vae_dir:
            vae_name = f'seed_{self.seed}_hardloss_{self.hard_loss}_batchloss_{self.batch_loss}_unsupervised_{self.unsupervised}_pseudo_{self.use_pseudo}_hardlatent_{self.latent_hard}_gumbelmax_{self.gumbel_max}'
            self.vae_id = lm_name

        if self.seq2seq_dir:
            seq2seq_name = f'seed_{self.seed}_hardloss_{self.hard_loss}_batchloss_{self.batch_loss}'
            self.seq2seq_id = seq2seq_name


