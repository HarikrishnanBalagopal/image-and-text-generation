"""
Train DCGAN for 256 x 256 images using pretrained weights from DCGAN for 64 x 64 images.
"""
# pylint: disable=wrong-import-order
# pylint: disable=ungrouped-imports
# The imports are ordered by length.

import os
import torch

from argparse import ArgumentParser
from imgtxtgen.arch1.models.dcgan import DCGAN256, train_dcgan
from imgtxtgen.common.utils import get_standard_img_transforms
from imgtxtgen.common.datasets.cub2011 import get_cub2011_data_loader

def parse_args():
    """
    Parse command line arguments.
    """
    # pylint: disable=bad-whitespace
    # The whitespace make it more readable.

    weights_path = os.path.join('outputs', 'arch1', 'generated_2020_01_09_03_06_33', 'weights', 'gen_epoch_200_iter_200.pth')
    outputs_path = os.path.join('outputs', 'arch1')

    parser = ArgumentParser()
    parser.add_argument('--dataset'    , type=str  , help='Path to CUB2011 directory.')
    parser.add_argument('--captions'   , type=str  , help='Path to captions directory.')
    parser.add_argument('--gpu'        , type=int  , help='ID of the GPU to use for training.'                     , default=0)
    parser.add_argument('--lr'         , type=float, help='Learning rate.'                                         , default=0.0002)
    parser.add_argument('--print_every', type=int  , help='Number of batches between saves.'                       , default=100)
    parser.add_argument('--batch'      , type=int  , help='Batch size.'                                            , default=16)
    parser.add_argument('--epochs'     , type=int  , help='Number of epochs to train for.'                         , default=400)
    parser.add_argument('--outputs'    , type=str  , help='Directory to store training outputs.'                   , default=outputs_path)
    parser.add_argument('--weights'    , type=str  , help='Path to DCGAN64 weights pretrained on CUB 2011 dataset.', default=weights_path)

    args = parser.parse_args()

    assert os.path.isfile(args.weights), f'weights path is not a file:{args.weights}'
    assert not args.dataset or os.path.isdir(args.dataset), f'CUB2011 path is not a directory:{args.dataset}'
    assert not args.captions or os.path.isdir(args.captions), f'Captions path is not a directory:{args.captions}'

    return args

def train_dcgan_256(args):
    """
    This train DCGAN256 on CUB 2011 dataset using weights from DCGAN64 model pretrained on the same dataset.
    """
    # pylint: disable=too-many-locals
    # pylint: disable=bad-whitespace
    # This number of local variables is necessary for training. The whitespace makes it more readable.

    d_gen         = 16
    d_dis         = 16
    d_noise       = 100
    d_image_size  = 256
    learning_rate = args.lr
    gpu_id        = args.gpu
    d_batch       = args.batch
    num_epochs    = args.epochs
    weights_path  = args.weights
    print_every   = args.print_every

    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
    print('training on device:', device)

    print('constructing DCGAN256 model:')
    model = DCGAN256(d_noise=d_noise, d_gen=d_gen, d_dis=d_dis).to(device)

    print('loading pretrained weights of dcgan 64:')
    pretrained_weights_64 = torch.load(weights_path)
    model.load_weights_from_dcgan_64(pretrained_weights_64)

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
    train_dcgan(model, data_loader,
                device=device,
                d_batch=d_batch,
                num_epochs=num_epochs,
                output_dir=args.outputs,
                print_every=print_every,
                config_name='dcgan_256_cub2011_using_pretrained_64',
                learning_rate=learning_rate)

if __name__ == '__main__':
    train_dcgan_256(parse_args())
