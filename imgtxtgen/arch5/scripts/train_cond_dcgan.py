"""
Train DCGAN for 256 x 256 images using pretrained weights from DCGAN for 64 x 64 images.
"""
# pylint: disable=wrong-import-order
# pylint: disable=ungrouped-imports
# The imports are ordered by length.

import os
import torch

from argparse import ArgumentParser
from imgtxtgen.common.utils import get_standard_img_transforms
from imgtxtgen.common.datasets.cub2011 import get_cub2011_data_loader
from imgtxtgen.arch5.models.cond_dcgan import CondDCGAN, train_cond_dcgan

def parse_args():
    """
    Parse command line arguments.
    """
    # pylint: disable=bad-whitespace
    # The whitespace make it more readable.

    outputs_path = os.path.join('outputs', 'arch5')

    parser = ArgumentParser()
    parser.add_argument('--dataset'    , type=str  , help='Path to CUB2011 directory.')
    parser.add_argument('--captions'   , type=str  , help='Path to captions directory.')
    parser.add_argument('--gpu'        , type=int  , help='ID of the GPU to use for training.'                     , default=0)
    parser.add_argument('--lr'         , type=float, help='Learning rate.'                                         , default=0.0002)
    parser.add_argument('--print_every', type=int  , help='Number of batches between saves.'                       , default=100)
    parser.add_argument('--batch'      , type=int  , help='Batch size.'                                            , default=16)
    parser.add_argument('--epochs'     , type=int  , help='Number of epochs to train for.'                         , default=400)
    parser.add_argument('--outputs'    , type=str  , help='Directory to store training outputs.'                   , default=outputs_path)

    args = parser.parse_args()

    assert not args.dataset or os.path.isdir(args.dataset), f'CUB2011 path is not a directory:{args.dataset}'
    assert not args.captions or os.path.isdir(args.captions), f'Captions path is not a directory:{args.captions}'

    return args

def train(args):
    """
    This trains CondDCGAN on 64 x 64 images from CUB 2011 dataset.
    """
    # pylint: disable=too-many-locals
    # pylint: disable=bad-whitespace
    # This number of local variables is necessary for training. The whitespace makes it more readable.

    d_gen         = 16
    d_dis         = 16
    d_noise       = 100
    d_image_size  = 64 # currently the network is designed for 64 x 64 color images.
    learning_rate = args.lr
    gpu_id        = args.gpu
    d_batch       = args.batch
    num_epochs    = args.epochs
    print_every   = args.print_every
    output_dir    = args.outputs
    d_classes     = 200 # for CUB 2011.
    d_embed       = 256
    d_hidden      = 256

    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
    print('training on device:', device)

    print('constructing CondDCGAN model:')
    model = CondDCGAN(d_noise=d_noise, d_classes=d_classes, d_embed=d_embed, d_hidden=d_hidden, d_gen=d_gen, d_dis=d_dis).to(device)

    print('loading cub2011:')
    img_transforms = get_standard_img_transforms(d_image_size)

    dataset_opts = {'img_transforms': img_transforms}
    if args.dataset:
        dataset_opts.dataset_dir = args.dataset
    if args.captions:
        dataset_opts.captions_dir = args.captions

    data_loader, _ = get_cub2011_data_loader(d_batch=d_batch, **dataset_opts)

    print('training config:', args)

    print('training on cub2011:')
    train_cond_dcgan(model=model,
                     data_loader=data_loader,
                     device=device,
                     d_batch=d_batch,
                     num_epochs=num_epochs,
                     output_dir=output_dir,
                     print_every=print_every,
                     learning_rate=learning_rate)

if __name__ == '__main__':
    train(parse_args())
