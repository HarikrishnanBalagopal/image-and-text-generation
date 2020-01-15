"""
Network to encode images and get features.
Model architecture taken from pytorch image captioning tutorial.
https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/03-advanced/image_captioning/model.py
"""

import torch

from torch import nn
from torchvision.models import resnet152

class ImageEncoder(nn.Module):
    """
    Encoder to get image features.
    """

    def __init__(self, d_embed):
        super().__init__()
        self.d_embed = d_embed
        self.define_module()

    def define_module(self):
        """
        Define each part of ImageEncoder.
        """
        # pylint: disable=invalid-name
        # Small names like fc abd bn are fine in this context.

        resnet = resnet152(pretrained=True)
        modules = list(resnet.children())[:-1] # delete the last fc layer.
        for module in modules:
            module.requires_grad = False
        self.resnet = nn.Sequential(*modules)
        self.fc = nn.Linear(resnet.fc.in_features, self.d_embed)
        self.bn = nn.BatchNorm1d(self.d_embed, momentum=0.01)

    def forward(self, images):
        """
        Run the network on images to get features.
        images : d_batch x 3 x d_image_size x d_image_size
        """
        # pylint: disable=arguments-differ
        # The arguments will differ from the base class since nn.Module is an abstract class.

        with torch.no_grad():
            latent = self.resnet(images).squeeze()
        features = self.bn(self.fc(latent))
        return features
