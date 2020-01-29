"""
Tests for flickr30k dataset.
"""

from imgtxtgen.common.datasets.flickr30k import get_default_flickr30k_loader

def test_flickr30k():
    """
    Tests for flickr30k dataset.
    """

    d_batch = 64
    d_image_size = 256

    dataset, train_loader = get_default_flickr30k_loader(d_batch=d_batch, split='train', d_image_size=d_image_size)
    _, val_loader = get_default_flickr30k_loader(d_batch=d_batch, split='val', d_image_size=d_image_size)
    _, test_loader = get_default_flickr30k_loader(d_batch=d_batch, split='test', d_image_size=d_image_size)

    batch = next(iter(train_loader))
    images, captions, cap_lens = batch

    assert images.size() == (d_batch, 3, d_image_size, d_image_size)
    assert captions.size(0) == d_batch
    assert cap_lens.size() == (d_batch,)

    batch = next(iter(val_loader))
    images, captions, cap_lens = batch

    assert images.size() == (d_batch, 3, d_image_size, d_image_size)
    assert captions.size(0) == d_batch
    assert cap_lens.size() == (d_batch,)

    batch = next(iter(test_loader))
    images, captions, cap_lens = batch

    assert images.size() == (d_batch, 3, d_image_size, d_image_size)
    assert captions.size(0) == d_batch
    assert cap_lens.size() == (d_batch,)

    assert dataset
    decoded = dataset.decode(captions[0])
    assert len(decoded) == len(captions[0])
