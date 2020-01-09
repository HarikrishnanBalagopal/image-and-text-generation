"""
Module for CUB2011 dataset with captions.
The images are from the paper Caltech-UCSD Birds-200-2011 https://authors.library.caltech.edu/27452/1/CUB_200_2011.pdf
The images can be downloaded here: http://www.vision.caltech.edu/visipedia/CUB-200-2011.html
The captions are from the paper Learning Deep Representations of Fine-Grained Visual Descriptions https://arxiv.org/pdf/1605.05395.pdf
The captions can be downloaded here: https://github.com/reedscot/cvpr2016
"""

import os
import re
import torch
import tarfile
import pandas as pd
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url

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
        return torch.tensor([self.word2idx[word] for word in words], dtype=torch.long)

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

    def __init__(self, dataset_dir, captions_dir, split='train', img_transforms=None, loader=default_loader):
        self.dataset_dir = os.path.expanduser(dataset_dir)
        assert os.path.isdir(self.dataset_dir), f'The dataset path {self.dataset_dir} is not a directory.'

        self.images_dir = os.path.join(self.dataset_dir, 'images')
        assert os.path.isdir(self.dataset_dir), f'The images path {self.images_dir} is not a directory.'

        self.captions_dir = os.path.join(captions_dir, 'text_c10')
        assert os.path.isdir(self.captions_dir), f'The captions path {self.captions_dir} is not a directory.'

        self.img_transforms = img_transforms
        self.loader = loader
        self.split = split
        self.dict = Dictionary()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.')

    def _load_metadata(self):
        """
        Load the metadata such as image filepaths, class names and whether image belongs to train or test split.
        """

        images             = pd.read_csv(os.path.join(self.dataset_dir, 'images.txt')            , sep=' ', names=['img_id', 'filepath'       ])
        image_class_labels = pd.read_csv(os.path.join(self.dataset_dir, 'image_class_labels.txt'), sep=' ', names=['img_id', 'target'         ])
        train_test_split   = pd.read_csv(os.path.join(self.dataset_dir, 'train_test_split.txt')  , sep=' ', names=['img_id', 'is_training_img'])

        self.data = images.merge(image_class_labels, on='img_id').merge(train_test_split, on='img_id')
        self.data = self.data[self.data.is_training_img == (1 if self.split == 'train' else 0)]


    def _load_captions(self):

        captions = []
        end_token = self.dict.add_word('<eos>')

        for _, row in self.data.iterrows():
            cap_filepath = os.path.join(self.captions_dir, row.filepath[:-3] + 'txt')

            if not os.path.isfile(cap_filepath):
                print(cap_filepath)
                return False

            with open(cap_filepath) as f:
                lines = f.read().strip().lower()

            lines = re.sub(r"([,.])", r" \1 ", lines)
            lines = re.sub(r"[^a-z\n,. ]+", ' ', lines).split('\n')
            current_image_captions = []

            for line in lines:
                words = line.strip().split()
                if len(words) == 0:
                    continue
                caption_list = [SOS_TOKEN] + [self.dict.add_word(word) for word in words] + [EOS_TOKEN]
                current_image_captions.append(torch.tensor(caption_list, dtype=torch.long))

            if len(current_image_captions) == 0:
                print('MUST HAVE ATLEAST 1 VALID CAPTION FOR EACH IMAGE')
                print('caption file path:', cap_filepath)
                return False

            captions.append(current_image_captions)
        self.captions = captions

    def _check_integrity(self):
        """
        Load images and captions and check integrity of the dataset.
        """

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
        path = os.path.join(self.root, self.base_folder, sample.filepath)
        target = sample.target - 1  # Targets start at 1 by default, so shift to 0
        img = self.loader(path)

        if self.transform is not None:
            img = self.transform(img)

        caps = self.captions[idx]
        return img, caps, target

def run_tests():
    print('reading CUB200-2011 dataset')
    trainset = CUB2011Dataset(dataset_dir'..', captions_dir='../cub_with_captions', download=False)
    
if __name__ == '__main__':
    run_tests()
