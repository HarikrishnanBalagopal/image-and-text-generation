"""
Module for celeba dataset stored in /users/gpu/haribala/code/datasets/celeba.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt

import torchvision
from torchvision.datasets import ImageFolder
from torchvision import transforms
_DATASET_FOLDER = '/users/gpu/haribala/code/datasets/celeba'
_IMAGE_SIZE = 64

CELEBA_DATASET = ImageFolder(root=_DATASET_FOLDER, transform=transforms.Compose([
    transforms.Resize(_IMAGE_SIZE),
    transforms.CenterCrop(_IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]))

def run_tests():
    """
    Run tests for Celeba dataset.
    """

    d_batch = 128
    dataloader = torch.utils.data.DataLoader(CELEBA_DATASET, batch_size=d_batch, shuffle=True, num_workers=4)

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
