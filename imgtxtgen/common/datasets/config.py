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

FLICKR30K_DATASET_PATH = os.path.join(DATASETS_PATH, 'flickr30k')
FLICKR30K_DATASET_IMAGE_FOLDER_PATH = os.path.join(FLICKR30K_DATASET_PATH, 'images')
FLICKR30K_DATASET_CAPTIONS_PATH = os.path.join(FLICKR30K_DATASET_PATH, 'captions.json')
FLICKR30K_DATASET_KARPATHY_SPLIT_AND_CAPTIONS_PATH = os.path.join(FLICKR30K_DATASET_PATH, 'karpathy_split_and_captions.json')
FLICKR30K_DATASET_IMAGES_PATH = os.path.join(FLICKR30K_DATASET_PATH, 'images', 'flickr30k-images')
