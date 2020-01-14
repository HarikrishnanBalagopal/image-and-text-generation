"""
Some configuration options such as paths to folders containing various datasets.
"""

import os

DATASETS_PATH = os.path.join('/', 'users', 'gpu', 'haribala', 'code', 'datasets')

CELEBA_DATASET_PATH = os.path.join(DATASETS_PATH, 'celeba')
CELEBA_DATASET_IMAGE_FOLDER_PATH = CELEBA_DATASET_PATH

CUB_200_2011_DATASET_PATH = os.path.join(DATASETS_PATH, 'CUB_200_2011', 'CUB_200_2011')
CUB_200_2011_DATASET_IMAGE_FOLDER_PATH = os.path.join(CUB_200_2011_DATASET_PATH, 'images')
CUB_200_2011_DATASET_CAPTIONS_PATH = os.path.join(DATASETS_PATH, 'CUB_200_2011_captions')
