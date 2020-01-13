"""
Module for the image captioning model.
Uses an image encoder to get local and global image features.
It then generates the caption with a LSTM using global attention on the image features.
"""
# pylint: disable=wrong-import-order
# pylint: disable=ungrouped-imports
# The imports are ordered by length.

import os
import torch
import datetime
import torch.nn as nn

from utils import mkdir_p
from torchvision import transforms
from .image_encoder import ImageEncoder
from datasets.cub2011 import CUB2011Dataset
from .lstm_with_attention import LSTMWithAttention

class ImageToText(nn.Module):
    """
    Generates a caption given an image.
    """
    # pylint: disable=too-many-instance-attributes
    # The attributes are necessary.

    def __init__(self, d_vocab, d_embed, d_annotations, d_hidden, d_max_seq_len, d_global_image_features, end_token=0, start_token=1):
        # pylint: disable=too-many-arguments
        # The arguments are necessary.

        super().__init__()
        self.d_vocab = d_vocab
        self.d_embed = d_embed
        self.d_annotations = d_annotations
        self.d_hidden = d_hidden
        self.d_max_seq_len = d_max_seq_len
        self.d_global_image_features = d_global_image_features
        self.end_token = end_token
        self.start_token = start_token
        self.define_module()

    def define_module(self):
        """
        Define each part of the model.
        """

        self.img_enc = ImageEncoder(d_embed=self.d_annotations)
        lstm_opts = {
            'd_vocab': self.d_vocab,
            'd_embed': self.d_embed,
            'd_annotations': self.d_annotations,
            'd_hidden': self.d_hidden,
            'd_max_seq_len': self.d_max_seq_len,
            'd_global_image_features': self.d_global_image_features,
            'end_token': self.end_token,
            'start_token': self.start_token
        }
        self.rnn = LSTMWithAttention(**lstm_opts)

    def forward(self, images):
        """
        Run the model on an image and generate the caption for it.
        """
        # pylint: disable=arguments-differ
        # The arguments will differ from the base class since nn.Module is an abstract class.

        image_features = self.img_enc(images)
        captions, captions_logits, attn_maps = self.rnn(image_features)
        return captions, captions_logits, attn_maps

def run_tests():
    """
    Run tests for ImageToText.
    """

    gpu_id = 1
    d_batch = 64 # works upto 32.
    num_epochs = 200
    print_every = 100
    d_image_size = 256
    d_max_seq_len = 18
    save_results = True
    learning_rate = 0.05

    timestamp = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    output_dir = os.path.join('.', 'output', f'img_to_text_{timestamp}')
    output_weights_dir = os.path.join(output_dir, 'weights')
    mkdir_p(output_weights_dir)

    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
    print('device', device)

    print('Loading cub2011 dataset:')
    cub2011_dataset_dir = '../../../exp2/CUB_200_2011'
    cub2011_captions_dir = '../../../exp2/cub_with_captions'
    img_transforms = transforms.Compose([
        transforms.Resize((d_image_size, d_image_size)),
        transforms.ToTensor(),
        transforms.Normalize((.5, .5, .5), (.5, .5, .5))
    ])
    cub2011_dataset = CUB2011Dataset(dataset_dir=cub2011_dataset_dir, captions_dir=cub2011_captions_dir, img_transforms=img_transforms, d_max_seq_len=d_max_seq_len)
    dataloader = torch.utils.data.DataLoader(cub2011_dataset, batch_size=d_batch, shuffle=True, drop_last=True, num_workers=4, pin_memory=True)

    d_vocab = len(cub2011_dataset.dict)
    print('d_vocab', d_vocab)
    image_to_text_opts = {
        'd_vocab': d_vocab,
        'd_embed': 256,
        'd_annotations': 256,
        'd_hidden': 256,
        'd_max_seq_len': 18,
        'd_global_image_features': 256,
        'end_token': cub2011_dataset.end_token,
        'start_token': cub2011_dataset.start_token
    }

    img_to_text = ImageToText(**image_to_text_opts).to(device)
    print(img_to_text)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(img_to_text.parameters(), lr=learning_rate)

    for epoch in range(1, num_epochs+1):
        for i, batch in enumerate(dataloader, start=1):
            imgs, captions, _ = batch
            imgs, captions = imgs.to(device), captions.to(device)
            pred_captions, pred_captions_logits, _ = img_to_text(imgs)
            loss = loss_fn(pred_captions_logits, captions)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % print_every == 0:
                print('epoch', epoch, 'i:', i)
                print('captions[0]:', cub2011_dataset.dict.decode(captions[0]))
                print('pred_captions[0]:', cub2011_dataset.dict.decode(pred_captions[0]))
                print('loss:', loss)
                if save_results:
                    print(f'saving results to {output_weights_dir}:')
                    torch.save(img_to_text.state_dict(), os.path.join(output_weights_dir, f'img_to_text_epoch_{epoch}_iter_{i}.pth'))

if __name__ == '__main__':
    run_tests()
