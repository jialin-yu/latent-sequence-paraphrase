from trainer import Trainer
from config import Configs
import argparse
import torch


def main():
    '''
    if batch_loss: penalty on long sequence; if not: sequence length does not matter
    if hard: use catergorical representation; if not: use smooth representation 
    '''

    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d', '--data', type=str, default='mscoco')
    parser.add_argument(
        '-utrs', '--un_train_size', type=int, default=20000)
    parser.add_argument(
        '-trs', '--train_size', type=int, default=10000)
    
    parser.add_argument(
        '-lh', '--latent_hard', type=str2bool, default=False)
    
    parser.add_argument(
        '-gum', '--gumbel_max', type=str2bool, default=True)
    parser.add_argument(
        '-ft', '--fixed_temperature', type=str2bool, default=False)
    
    parser.add_argument(
        '-plm', '--use_pretrain_lm', type=str2bool, default=True)

    parser.add_argument(
        '-up', '--use_pseudo', type=str2bool, default=False)

    # LM experiment
    parser.add_argument(
        '-lmdir', '--lm_dir', type=str, default='../model/lm/')
    parser.add_argument(
        '-lmme', '--lm_max_epoch', type=int, default=15)
    parser.add_argument(
        '-lmlr', '--lm_lr', type=int, default=1e-4)

    parser.add_argument(
        '-vaedir', '--vae_dir', type=str, default='../model/vae/')
    parser.add_argument(
        '-vaeme', '--vae_max_epoch', type=int, default=15)
    parser.add_argument(
        '-vaelr', '--vae_lr', type=int, default=1e-4)

    parser.add_argument(
        '-seq2seqdir', '--seq2seq_dir', type=str, default='../model/seq2seq/')
    parser.add_argument(
        '-seq2seqme', '--seq2seq_max_epoch', type=int, default=2)
    parser.add_argument(
        '-seq2seqlr', '--seq2seq_lr', type=int, default=1e-5)
    parser.add_argument(
        '-duo_train', '--duo', type=str2bool, default=True)

    parser.add_argument(
        '-semidir', '--semi_dir', type=str, default='../model/semi/')
    parser.add_argument(
        '-semime', '--semi_max_epoch', type=int, default=15)
    parser.add_argument(
        '-semilr', '--semi_lr', type=int, default=1e-4)
    
    
    
    parser.add_argument(
        '-hd', '--hid_dim', type=int, default=512)
    parser.add_argument(
        '-nh', '--n_heads', type=int, default=8)
    parser.add_argument(
        '-nl', '--n_lays', type=int, default=6)
    parser.add_argument(
        '-dp', '--dropout', type=int, default=0.1)

    parser.add_argument(
        '-s', '--seed', type=int, default=1234)

    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()

    configs = Configs(**vars(args))
    interface = Trainer(configs)
    interface.main_lm()
    # interface.main_vae()
    # interface.main_seq2seq()
    interface.main_semi_supervised()


if __name__ == "__main__":
    main()