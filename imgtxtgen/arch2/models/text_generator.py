"""
Generator part of Img2TextGAN.
"""
import torch

from torch import nn
from imgtxtgen.arch2.models.image_encoder import ImageEncoder
from imgtxtgen.arch2.models.recursive_lstm import RecursiveLSTM

class TextGenerator(nn.Module):
    """
    Takes images and some noise to generates captions.
    """

    def __init__(self, d_vocab, d_embed, d_hidden, d_max_seq_len, d_image_features, d_noise, end_token=0, start_token=1):
        super().__init__()
        self.d_vocab = d_vocab
        self.d_embed = d_embed
        self.d_noise = d_noise
        self.d_hidden = d_hidden
        self.end_token = end_token
        self.start_token = start_token
        self.d_max_seq_len = d_max_seq_len
        self.d_image_features = d_image_features
        self.d_condition = self.d_image_features + self.d_noise

        self.define_module()

    def define_module(self):
        """
        Define each part of the TextGenerator model.
        """
        self.img_enc = ImageEncoder(d_embed=self.d_image_features)
        self.text_gen = RecursiveLSTM(d_vocab=self.d_vocab, d_embed=self.d_embed, d_hidden=self.d_hidden, d_max_seq_len=self.d_max_seq_len, d_condition=self.d_condition, end_token=self.end_token, start_token=self.start_token)

    def forward(self, images, noise):
        """
        Run the network forward.
        """
        # pylint: disable=arguments-differ
        # The arguments will differ from the base class since nn.Module is an abstract class.

        features = self.img_enc(images)
        latent = torch.cat((features, noise), dim=1)
        captions, captions_log_probs, hiddens = self.text_gen(latent)
        return captions, captions_log_probs, hiddens
