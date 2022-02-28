from trainer import Trainer
from config import Configs
import argparse
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d', '--data', type=str, default='quora')
    parser.add_argument(
        '-e', '--unsupervised', type=bool, default=True)
    parser.add_argument(
        '-mv', '--max_vocab', type=int, default=20000)
    parser.add_argument(
        '-trs', '--train_size', type=int, default=1000)
    parser.add_argument(
        '-vs', '--valid_size', type=int, default=50)
    parser.add_argument(
        '-ts', '--test_size', type=int, default=200)
    # LM experiment
    parser.add_argument(
        '-lmdir', '--lm_dir', type=str, default='../model/lm/')
    parser.add_argument(
        '-lmme', '--lm_max_epoch', type=int, default=10)
    parser.add_argument(
        '-lmlr', '--lm_lr', type=int, default=1e-2)
    parser.add_argument(
        '-gc', '--lm_gc', type=int, default=0.01)
    parser.add_argument(
        '-lmh', '--lm_hard', type=bool, default=True)
    parser.add_argument(
        '-hd', '--hid_dim', type=int, default=512)
    parser.add_argument(
        '-nh', '--n_heads', type=int, default=8)
    parser.add_argument(
        '-nl', '--n_lays', type=int, default=1)
    parser.add_argument(
        '-dp', '--dropout', type=int, default=0.1)

    parser.add_argument(
        '-s', '--seed', type=int, default=1234)

    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()

    configs = Configs(**vars(args))
    interface = Trainer(configs)
    interface.main_lm()


if __name__ == "__main__":
    main()