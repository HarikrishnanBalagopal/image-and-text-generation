"""
Flickr30k dataset.
Images and captions.
"""
# pylint: disable=wrong-import-order
# pylint: disable=ungrouped-imports
# The imports are ordered by length.

import os
import json
import torch

from collections import Counter
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.datasets.folder import default_loader
from imgtxtgen.common.utils import get_standard_img_transforms
from imgtxtgen.common.datasets.config import FLICKR30K_DATASET_IMAGES_PATH, FLICKR30K_DATASET_KARPATHY_SPLIT_AND_CAPTIONS_PATH

def collate_fn(data):
    """
    Creates mini-batch tensors from the list of tuples (image, caption).
    Args:
        data: list of tuple (image, caption).
            - image: torch float tensor of shape (3, d_image_size, d_image_size).
            - caption: torch long tensor of shape (?); variable length.
    Returns:
        images: torch float tensor of shape (d_batch, 3, d_image_size, d_image_size).
        captions: torch long tensor of shape (d_batch, padded_length).
        cap_lens: list; valid length for each padded caption.
    """
    # Sort a data list by caption length (descending order).
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, caps = zip(*data)

    # Merge images (from tuple of 3D tensor to 4D tensor).
    images = torch.stack(images)

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    cap_lens = torch.LongTensor([len(cap) for cap in caps])
    captions = torch.zeros(len(caps), max(cap_lens), dtype=torch.long)

    for i, cap in enumerate(caps):
        captions[i, :cap_lens[i]] = torch.LongTensor(cap)

    return images, captions, cap_lens

class Flickr30k(Dataset):
    """
    Flickr30k dataset.
    Only one of d_image_size or img_transforms should be provided.
    img_transforms overrides d_image_size if present.
    """

    def __init__(self, img_dir=FLICKR30K_DATASET_IMAGES_PATH, caps_path=FLICKR30K_DATASET_KARPATHY_SPLIT_AND_CAPTIONS_PATH, d_image_size=64, img_transforms=None, split='train', min_word_freq=5, img_loader=default_loader):
        self.split = split
        self.img_dir = img_dir
        self.caps_path = caps_path
        self.img_loader = img_loader
        self.d_image_size = d_image_size
        self.min_word_freq = min_word_freq
        self.img_transforms = get_standard_img_transforms(d_image_size) if img_transforms is None else img_transforms

        self.build_vocab()
        self.load_img_paths_and_captions()

    def build_vocab(self):
        """
        Build word to index mapping using all captions regardless of split.
        """
        # pylint: disable=bad-whitespace
        # The whitespace makes the code more readable.

        with open(self.caps_path, 'r') as caps_file:
            cap_data = json.load(caps_file)

        word_freq = Counter()
        images = cap_data['images']
        for image in images:
            for img_caption in image['sentences']:
                word_freq.update(img_caption['tokens'])

        # Create word to index mapping
        min_word_freq       = self.min_word_freq
        words               = [w for w in word_freq.keys() if word_freq[w] > min_word_freq]
        word2idx            = {word: idx for idx, word in enumerate(words, start=1)}
        word2idx['<pad>']   = 0
        word2idx['<unk>']   = len(word2idx)
        word2idx['<start>'] = len(word2idx)
        word2idx['<end>']   = len(word2idx)

        self.word2idx    = word2idx
        self.idx2word    = {idx: word for word, idx in word2idx.items()}
        self.d_vocab     = len(word2idx)
        self.pad_token   = word2idx['<pad>']
        self.unk_token   = word2idx['<unk>']
        self.start_token = word2idx['<start>']
        self.end_token   = word2idx['<end>']

    def encode(self, caption):
        """
        Encode a caption.
        """

        if isinstance(caption, torch.Tensor):
            caption = caption.tolist() # iterating over a 1d tensor produces scalar tensors.
        word2idx = self.word2idx
        unk_token = self.unk_token
        caption = [word2idx.get(word, unk_token) for word in caption]
        return caption

    def decode(self, caption):
        """
        Decode a caption.
        """

        if isinstance(caption, torch.Tensor):
            caption = caption.tolist() # iterating over a 1d tensor produces scalar tensors.
        idx2word = self.idx2word
        caption = [idx2word[idx] for idx in caption]
        return caption

    def load_img_paths_and_captions(self):
        """
        Load the image paths and captions.
        """

        with open(self.caps_path, 'r') as caps_file:
            cap_data = json.load(caps_file)

        img_paths = []
        captions = []

        images = cap_data['images']
        start_token = self.start_token
        end_token = self.end_token
        for image in images:
            if image['split'] != self.split:
                continue
            img_paths.append(image['filename'])
            img_captions = []
            for img_caption in image['sentences']:
                img_captions.append([start_token] + self.encode(img_caption['tokens']) + [end_token])
            captions.append(img_captions)

        # Sanity check
        assert len(img_paths) == len(captions)

        self.img_paths = img_paths
        self.captions = captions

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = self.img_loader(os.path.join(self.img_dir, self.img_paths[idx]))
        img = self.img_transforms(img)
        captions = self.captions[idx]
        caption = captions[torch.randint(low=0, high=len(captions), size=(1,)).item()]
        return img, caption

def get_default_flickr30k_loader(d_batch, **kwargs):
    """
    Returns the dataset and the default loader for Flickr30k.
    :return: (dataset, data_loader)
    """

    dataset = Flickr30k(**kwargs)
    data_loader = DataLoader(dataset, batch_size=d_batch, num_workers=4, shuffle=True, pin_memory=True, collate_fn=collate_fn)
    return dataset, data_loader
