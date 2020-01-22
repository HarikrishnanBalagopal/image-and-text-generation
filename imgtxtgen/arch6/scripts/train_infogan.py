"""
Train InfoGAN on MNIST.
"""

import torch

from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from imgtxtgen.arch6.models.infogan import InfoGAN, train

def train_infogan():
    """
    Train InfoGAN.
    """

    gpu_id = 1
    d_batch = 64
    num_epochs = 100
    print_every = 100

    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')

    img_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    datasets_dir = '/users/gpu/haribala/code/datasets'
    mnist = MNIST(datasets_dir, train=True, download=True, transform=img_transforms)
    train_loader = DataLoader(mnist, batch_size=d_batch, shuffle=True, num_workers=4)

    model = InfoGAN().to(device)
    train(model, train_loader, device, d_batch, num_epochs=num_epochs, print_every=print_every)

if __name__ == '__main__':
    train_infogan()
