"""
Module for the image captioning model.
Uses an image encoder to get local and global image features.
It then generates the caption with a LSTM using global attention on the image features.
"""

import torch
import torch.nn as nn
from image_encoder import ImageEncoder
from lstm_with_attention import LSTMWithAttention

class ImageToText(nn.Module):
    """
    Generates a caption given an image.
    """
    # pylint: disable=too-many-instance-attributes
    # The attributes are necessary.

    def __init__(self, d_vocab, d_embed, d_annotations, d_hidden, d_max_seq_len, d_global_image_features):
        # pylint: disable=too-many-arguments
        # The arguments are necessary.

        super().__init__()
        self.d_vocab = d_vocab
        self.d_embed = d_embed
        self.d_annotations = d_annotations
        self.d_hidden = d_hidden
        self.d_max_seq_len = d_max_seq_len
        self.d_global_image_features = d_global_image_features
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
            'd_global_image_features': self.d_global_image_features
        }
        self.rnn = LSTMWithAttention(**lstm_opts)

    def forward(self, x):
        """
        Run the model on an image and generate the caption for it.
        """
        # pylint: disable=arguments-differ
        # The arguments will differ from the base class since nn.Module is an abstract class.
        print(x.size())

def run_tests():
    """
    Run tests.
    """

    d_batch = 20
    image_to_text_opts = {
        'd_vocab': 5450,
        'd_embed': 256,
        'd_annotations': 256,
        'd_hidden': 256,
        'd_max_seq_len': 18,
        'd_global_image_features': 256
    }
    img_to_text = ImageToText(**image_to_text_opts)
    print(img_to_text)
    s_images = (d_batch, 256, 256, 3)
    images = torch.randn(*s_images)
    print(images.size())

if __name__ == '__main__':
    run_tests()
