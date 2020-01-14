"""
Module for CUB2011 dataset with captions.
The images are from the paper Caltech-UCSD Birds-200-2011 https://authors.library.caltech.edu/27452/1/CUB_200_2011.pdf
The images can be downloaded here: http://www.vision.caltech.edu/visipedia/CUB-200-2011.html
The captions are from the paper Learning Deep Representations of Fine-Grained Visual Descriptions https://arxiv.org/pdf/1605.05395.pdf
The captions can be downloaded here: https://github.com/reedscot/cvpr2016
"""
# pylint: disable=wrong-import-order
# pylint: disable=ungrouped-imports
# The imports are ordered by length.

import os
import re
import torch
import tarfile
import pandas as pd

from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url
from torchvision.datasets.folder import default_loader
from imgtxtgen.common.datasets.config import CUB_200_2011_DATASET_PATH, CUB_200_2011_DATASET_CAPTIONS_PATH

def download_images(download_dir):
    """
    Download the CUB2011 dataset images. Captions are NOT downloaded.
    """

    url = 'http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz'
    filename = 'CUB_200_2011.tgz'
    tgz_md5 = '97eceeb196236b17998738112f37df78'

    download_url(url, download_dir, filename, tgz_md5)

    with tarfile.open(os.path.join(download_dir, filename), "r:gz") as tar:
        tar.extractall(path=download_dir)

class Dictionary:
    """
    Class to hold the mapping between words and integers.
    """

    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        """
        If the word is not present in the dictionary, add it.
        Either way return the idx of the word.
        """

        if word not in self.word2idx:
            self.word2idx[word] = len(self.idx2word)
            self.idx2word.append(word)
        return self.word2idx[word]

    def encode(self, sentence):
        """
        Encode a sentence into a tensor of word indices.
        """

        words = sentence.strip().lower().split()
        return torch.LongTensor([self.word2idx[word] for word in words])

    def decode(self, idxs):
        """
        Decode a tensor of word indices into a sentence.
        """

        return ' '.join(self.idx2word[idx] for idx in idxs)

    def __len__(self):
        return len(self.idx2word)

class CUB2011Dataset(Dataset):
    """
    Class to load the CUB2011 dataset with captions.
    """
    # pylint: disable=too-many-instance-attributes
    # The attributes are necessary for this dataset.

    def __init__(self, dataset_dir=CUB_200_2011_DATASET_PATH, captions_dir=CUB_200_2011_DATASET_CAPTIONS_PATH, split='train', d_max_seq_len=18, img_transforms=None, loader=default_loader):
        # pylint: disable=too-many-arguments
        # The arguments are necessary to construct this dataset.

        super().__init__()
        self.dataset_dir = os.path.expanduser(dataset_dir)
        self.images_dir = os.path.join(self.dataset_dir, 'images')
        self.captions_dir = captions_dir

        assert os.path.isdir(self.dataset_dir), f'The dataset path {self.dataset_dir} is not a directory.'
        assert os.path.isdir(self.dataset_dir), f'The images path {self.images_dir} is not a directory.'
        assert os.path.isdir(self.captions_dir), f'The captions path {self.captions_dir} is not a directory.'

        self.d_max_seq_len = d_max_seq_len
        self.img_transforms = img_transforms
        self.loader = loader
        self.split = split
        self.dict = Dictionary()
        self.end_token = self.dict.add_word('<eos>')
        self.start_token = self.dict.add_word('<sos>')

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.')

    def _load_metadata(self):
        """
        Load the metadata such as image filepaths, class names and whether image belongs to train or test split.
        """
        # pylint: disable=attribute-defined-outside-init
        # pylint: disable=bad-whitespace
        # This method is called by init. The whitespace makes the code more readable.

        images             = pd.read_csv(os.path.join(self.dataset_dir, 'images.txt')            , sep=' ', names=['img_id', 'filepath'       ])
        image_class_labels = pd.read_csv(os.path.join(self.dataset_dir, 'image_class_labels.txt'), sep=' ', names=['img_id', 'target'         ])
        train_test_split   = pd.read_csv(os.path.join(self.dataset_dir, 'train_test_split.txt')  , sep=' ', names=['img_id', 'is_training_img'])

        self.data = images.merge(image_class_labels, on='img_id').merge(train_test_split, on='img_id')
        self.data = self.data[self.data.is_training_img == (1 if self.split == 'train' else 0)]


    def _load_captions(self):
        """
        Load the captions.
        """
        # pylint: disable=attribute-defined-outside-init
        # This method is called by init.

        self.captions = []

        for _, row in self.data.iterrows():
            cap_filepath = os.path.join(self.captions_dir, row.filepath[:-3] + 'txt')

            if not os.path.isfile(cap_filepath):
                raise RuntimeError(f'This caption path {cap_filepath} is not a file.')

            with open(cap_filepath) as cap_file:
                lines = cap_file.read().strip().lower()

            lines = re.sub(r"([,.])", r" \1 ", lines) # Add a space before and after every comma and dot.
            lines = re.sub(r"[^a-z\n,. ]+", ' ', lines).split('\n') # Replace long stretches of invalid characters with space.

            current_image_captions = []

            for line in lines:
                words = line.strip().split()[:self.d_max_seq_len - 1]
                if len(words) == 0:
                    continue
                num_padding = self.d_max_seq_len - len(words)
                caption = [self.dict.add_word(word) for word in words] + num_padding * [self.end_token]
                current_image_captions.append(torch.LongTensor(caption))

            if len(current_image_captions) == 0:
                raise RuntimeError(f'Must have atleast 1 valid caption for each image, caption file path: {cap_filepath}')

            self.captions.append(current_image_captions)

    def _check_integrity(self):
        """
        Load images and captions and check integrity of the dataset.
        """
        # pylint: disable=broad-except
        # This integrity check should fail if ANY exception is thrown.

        try:
            self._load_metadata()
            self._load_captions()
        except Exception:
            return False

        return True

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]

        img_filepath = os.path.join(self.images_dir, sample.filepath)
        img = self.loader(img_filepath)
        if self.img_transforms is not None:
            img = self.img_transforms(img)

        captions = self.captions[idx]
        caption = captions[torch.randint(low=0, high=len(captions), size=(1,)).item()]

        class_id = sample.target - 1  # Make class ids start at 0 (starts at 1 by default)

        return img, caption, class_id
