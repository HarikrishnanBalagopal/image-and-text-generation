"""
CUB2011 version.
Train InfoGAN on MNIST.
"""

import torch

from imgtxtgen.common.utils import get_standard_img_transforms
from imgtxtgen.arch6.models.infogan_cub2011 import InfoGAN, train
from imgtxtgen.common.datasets.cub2011 import get_cub2011_data_loader

def train_infogan():
    """
    Train InfoGAN.
    """

    gpu_id = 1
    d_batch = 64
    num_epochs = 100
    print_every = 25
    d_image_size = 256

    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')

    img_transforms = get_standard_img_transforms(d_image_size=d_image_size)
    train_loader, _ = get_cub2011_data_loader(d_batch=d_batch, img_transforms=img_transforms)

    model = InfoGAN().to(device)
    train(model, train_loader, device, d_batch, num_epochs=num_epochs, print_every=print_every)

if __name__ == '__main__':
    train_infogan()
