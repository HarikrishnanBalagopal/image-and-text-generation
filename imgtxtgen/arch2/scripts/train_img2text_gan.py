"""
Train Img2TextGAN.
"""

import torch

from torch.utils.data import DataLoader
from imgtxtgen.common.datasets.cub2011 import CUB2011Dataset
from imgtxtgen.common.utils import get_standard_img_transforms
from imgtxtgen.arch2.models.img2text_gan import Img2TextGAN, train_img2txt_gen

def train():
    """
    Train Img2TextGAN using reinforcement learning on the CUB2011 dataset.
    """
    # pylint: disable=bad-whitespace
    # The whitespace makes it more readable.

    d_embed          = 256
    d_hidden         = 256
    d_image_features = 2048
    d_noise          = 100
    d_max_seq_len    = 18
    d_image_size     = 256
    gpu_id           = 1
    d_batch          = 20
    num_epochs       = 100

    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
    print('training on device:', device)

    print('loading cub2011:')
    img_transforms = get_standard_img_transforms(d_image_size)

    dataset_opts = {'img_transforms': img_transforms, 'd_max_seq_len': d_max_seq_len}
    cub2011_dataset = CUB2011Dataset(**dataset_opts)
    dataloader = DataLoader(cub2011_dataset, batch_size=d_batch, shuffle=True, drop_last=True, num_workers=4, pin_memory=True)
    d_vocab = len(cub2011_dataset.dict)

    print('constructing Img2TextGAN model:')
    model = Img2TextGAN(d_vocab, d_embed, d_hidden, d_max_seq_len, d_image_features, d_noise, end_token=0, start_token=1).to(device)

    train_img2txt_gen(model, dataloader, d_batch, device, num_rollouts=16, num_epochs=num_epochs)

if __name__ == '__main__':
    train()
