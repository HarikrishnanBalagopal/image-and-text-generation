"""
Module for celeba dataset stored in /users/gpu/haribala/code/datasets/celeba.
Images only.
"""
# pylint: disable=wrong-import-order
# pylint: disable=ungrouped-imports
# The imports are ordered by length.

from torchvision import transforms
from torchvision.datasets import ImageFolder
from imgtxtgen.common.datasets.config import CELEBA_DATASET_IMAGE_FOLDER_PATH

def get_celeba_dataset(dataset_dir=CELEBA_DATASET_IMAGE_FOLDER_PATH, d_image_size=64):
    """
    Load the celeba dataset and resize images to a given size.
    """

    return ImageFolder(root=dataset_dir, transform=transforms.Compose([
        transforms.Resize((d_image_size, d_image_size)),
        transforms.CenterCrop(d_image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]))
