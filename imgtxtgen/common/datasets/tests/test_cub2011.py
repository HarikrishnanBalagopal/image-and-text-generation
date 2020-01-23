"""
Tests for CUB2011 dataset.
"""

from imgtxtgen.common.utils import get_standard_img_transforms
from imgtxtgen.common.datasets.cub2011 import get_cub2011_data_loader

def test_get_cub2011_data_loader():
    """
    Test the default data loader for CUB 2011 dataset.
    """
    d_batch = 20
    d_image_size = 64

    train_loader, dataset = get_cub2011_data_loader()
    images, captions, lengths, class_ids = next(iter(train_loader))

    assert dataset
    assert images.size() == (d_batch, 3, d_image_size, d_image_size)
    assert captions.size(0) == d_batch
    assert lengths.size() == (d_batch,)
    assert class_ids.size() == (d_batch,)

def test_get_cub2011_data_loader_256():
    """
    Test the default data loader for CUB 2011 dataset.
    """
    d_batch = 64
    d_image_size = 256

    img_transforms = get_standard_img_transforms(d_image_size=d_image_size)
    train_loader, dataset = get_cub2011_data_loader(d_batch=d_batch, img_transforms=img_transforms)
    images, captions, lengths, class_ids = next(iter(train_loader))

    assert dataset
    assert images.size() == (d_batch, 3, d_image_size, d_image_size)
    assert captions.size(0) == d_batch
    assert lengths.size() == (d_batch,)
    assert class_ids.size() == (d_batch,)
