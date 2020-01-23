"""
Tests for CelebA dataset.
"""

from torch.utils.data import DataLoader
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
