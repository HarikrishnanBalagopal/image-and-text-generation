"""
Module for celeba dataset stored in /users/gpu/haribala/code/datasets/celeba.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt

import torchvision
from torchvision.datasets import ImageFolder
from torchvision import transforms

def get_celeba_dataset(d_image_size):
    """
    Load the celeba dataset and resize images to a given size.
    """

    dataset_folder = '/users/gpu/haribala/code/datasets/celeba'

    return ImageFolder(root=dataset_folder, transform=transforms.Compose([
        transforms.Resize(d_image_size),
        transforms.CenterCrop(d_image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]))

def run_tests():
    """
    Run tests for Celeba dataset.
    """

    d_batch = 128
    celeba_dataset = get_celeba_dataset(64)
    dataloader = torch.utils.data.DataLoader(celeba_dataset, batch_size=d_batch, shuffle=True, num_workers=4)

    images, _ = next(iter(dataloader))
    image_grid = torchvision.utils.make_grid(images[:64], padding=2, normalize=True)
    image_grid = np.transpose(image_grid, (1, 2, 0)) # convert channels first to channels last format.

    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(image_grid)
    plt.savefig('test_batch_from_celeba_dataset.png')

if __name__ == '__main__':
    run_tests()
