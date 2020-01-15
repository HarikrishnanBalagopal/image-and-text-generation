"""
Tests for CUB2011 dataset.
"""

from imgtxtgen.common.datasets.cub2011 import get_cub2011_data_loader

def test_get_cub2011_data_loader():
    """
    Test the default data loader for CUB 2011 dataset.
    """

    d_batch = 20
    d_image_size = 64
    data_loader = get_cub2011_data_loader()

    images, captions, class_ids = next(iter(data_loader))

    assert images.size() == (d_batch, 3, d_image_size, d_image_size)
    assert captions.size(0) == d_batch
    assert class_ids.size() == (d_batch,)
