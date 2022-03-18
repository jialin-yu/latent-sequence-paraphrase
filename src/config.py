
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
            lm_id = f'data_{self.data}_trainsize_{self.un_train_size}_lr_{self.lm_lr}_seed_{self.seed}_hardloss_{self.hard_loss}_pseudo_{self.use_pseudo}_epoch_{self.lm_max_epoch}'
            self.lm_id = lm_id

        if self.vae_dir:
            vae_id = f'data_{self.data}_seed_{self.seed}_hardloss_{self.hard_loss}_pseudo_{self.use_pseudo}_hardlatent_{self.latent_hard}_gumbelmax_{self.gumbel_max}_epoch_{self.vae_max_epoch}'
            self.vae_id = vae_id

        if self.seq2seq_dir:
            seq2seq_id = f'data_{self.data}_trainsize_{self.train_size}_lr_{self.seq2seq_lr}_seed_{self.seed}_epoch_{self.seq2seq_max_epoch}'
            self.seq2seq_id = seq2seq_id

        if self.semi_dir:
            semi_id = f'data_{self.data}_untrainsize_{self.un_train_size}_trainsize_{self.train_size}_lr_{self.seq2seq_lr}_seed_{self.seed}_hardloss_{self.hard_loss}_epoch_{self.seq2seq_max_epoch}'
            self.semi_id = semi_id

