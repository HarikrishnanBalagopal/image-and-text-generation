"""
Tests for CUB2011 dataset.
"""

from torchvision import transforms
from torch.utils.data import DataLoader

from imgtxtgen.common.datasets.cub2011 import CUB2011Dataset

def test_cub2011():
    """
    Run tests for CUB2011Dataset.
    """

    d_batch = 20
    d_image_size = 64
    d_max_seq_len = 18

    img_transforms = transforms.Compose([
        transforms.Resize((d_image_size, d_image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    cub2011_dataset = CUB2011Dataset(img_transforms=img_transforms, d_max_seq_len=d_max_seq_len)
    dataloader = DataLoader(cub2011_dataset, batch_size=d_batch, shuffle=True, num_workers=4)

    images, captions, class_ids = next(iter(dataloader))

    assert images.size() == (d_batch, 3, d_image_size, d_image_size)
    assert captions.size() == (d_batch, d_max_seq_len)
    assert class_ids.size() == (d_batch,)
