"""
Tests for CelebA dataset.
"""

import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from imgtxtgen.common.datasets.celeba import get_celeba_dataset

def test_celeba():
    """
    Run tests for CelebA dataset.
    """

    d_batch = 128
    d_image_size = 256

    celeba_dataset = get_celeba_dataset(d_image_size=d_image_size)
    dataloader = DataLoader(celeba_dataset, batch_size=d_batch, shuffle=True, num_workers=4)

    images, _ = next(iter(dataloader))
    assert images.size() == (d_batch, 3, d_image_size, d_image_size)

    image_grid = make_grid(images[:64], padding=2, normalize=True)
    image_grid = np.transpose(image_grid, (1, 2, 0)) # convert channels first to channels last format.

    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title("CelebA training images.")
    plt.imshow(image_grid)
    plt.savefig('test_batch_from_celeba_dataset.png')
