"""
Module for utility functions.
"""

import os
import errno
import datetime

from torchvision import transforms

def mkdir_p(path):
    """
    Make directory and all necessary parent directories given by the path.
    """

    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def get_timestamp():
    """
    Return the current date and time in year, month, day, hour, minute and second format
    """

    now = datetime.datetime.now()
    return now.strftime('%Y_%m_%d_%H_%M_%S')

def get_standard_img_transforms(d_image_size=64):
    """
    Transforms to resize, convert to tensor and normalize images.
    """

    return transforms.Compose([
        transforms.Resize((d_image_size, d_image_size)),
        transforms.ToTensor(),
        transforms.Normalize((.5, .5, .5), (.5, .5, .5))
    ])
