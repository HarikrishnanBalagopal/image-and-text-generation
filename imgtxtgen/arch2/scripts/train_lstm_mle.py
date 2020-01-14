"""
Train recursive lstm using MLE.
"""
# pylint: disable=wrong-import-order
# pylint: disable=ungrouped-imports
# The imports are ordered by length.

import os
import torch

from argparse import ArgumentParser
from torch.utils.data import DataLoader
from imgtxtgen.common.datasets.cub2011 import CUB2011Dataset
from imgtxtgen.common.utils import get_standard_img_transforms
from imgtxtgen.arch2.models.recursive_lstm import RecursiveLSTM

def parse_args():
    """
    Parse command line arguments.
    """
    # pylint: disable=bad-whitespace
    # The whitespace make it more readable.

    outputs_path = os.path.join('outputs', 'arch2')

    parser = ArgumentParser()
    parser.add_argument('--dataset'    , type=str, help='Path to CUB2011 directory.')
    parser.add_argument('--captions'   , type=str, help='Path to captions directory.')
    parser.add_argument('--gpu'        , type=int, help='ID of the GPU to use for training.'   , default=1)
    parser.add_argument('--print_every', type=int, help='Number of batches between saves.'     , default=10)
    parser.add_argument('--batch'      , type=int, help='Batch size.'                          , default=16)
    parser.add_argument('--epochs'     , type=int, help='Number of epochs to train for.'       , default=200)
    parser.add_argument('--outputs'    , type=str, help='Directory to store training outputs.' , default=outputs_path)

    args = parser.parse_args()

    assert not args.dataset or os.path.isdir(args.dataset), f'CUB2011 path is not a directory:{args.dataset}'
    assert not args.captions or os.path.isdir(args.captions), f'Captions path is not a directory:{args.captions}'

    return args

def train_lstm_mle(args):
    """
    This train DCGAN256 on CUB 2011 dataset using weights from DCGAN64 model pretrained on the same dataset.
    """
    # pylint: disable=too-many-locals
    # pylint: disable=bad-whitespace
    # This number of local variables is necessary for training. The whitespace makes it more readable.

    d_max_seq_len = 18
    d_image_size  = 256
    d_condition   = 2048
    d_embed       = 256
    d_hidden      = 256
    gpu_id        = args.gpu
    d_batch       = args.batch
    num_epochs    = args.epochs
    print_every   = args.print_every

    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
    print('training on device:', device)

    print('loading cub2011:')

    dataset_opts = {'img_transforms': get_standard_img_transforms(d_image_size), 'd_max_seq_len': d_max_seq_len}
    if args.dataset:
        dataset_opts.dataset_dir = args.dataset
    if args.captions:
        dataset_opts.captions_dir = args.captions

    cub2011_dataset = CUB2011Dataset(**dataset_opts)
    dataloader = DataLoader(cub2011_dataset, batch_size=d_batch, shuffle=True, drop_last=True, num_workers=4, pin_memory=True)
    d_vocab = len(cub2011_dataset.dict)

    print('constructing RecursiveLSTM model:')
    model = RecursiveLSTM(d_vocab=d_vocab, d_embed=d_embed, d_hidden=d_hidden, d_max_seq_len=d_max_seq_len, d_condition=d_condition, end_token=cub2011_dataset.end_token, start_token=cub2011_dataset.start_token).to(device)

    print('training on cub2011:')
    train_dcgan(model, dataloader, device=device, d_batch=d_batch, num_epochs=num_epochs, output_dir=args.outputs, print_every=print_every, config_name='dcgan_256_cub2011_using_pretrained_64')

if __name__ == '__main__':
    train_lstm_mle(parse_args())
