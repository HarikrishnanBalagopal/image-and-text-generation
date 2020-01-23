"""
CUB2011 version.
Generator for InfoGAN.
Architecture taken from the InfoGAN paper:
https://arxiv.org/pdf/1606.03657.pdf
Architecture code:
https://github.com/openai/InfoGAN/blob/master/infogan/models/regularized_gan.py
"""

import torch

from torch import nn

class Generator(nn.Module):
    """
    Generator of InfoGAN for CUB2011.
    Built for MNIST dataset: 28 x 28 grayscale images from 10 classes.
    """

    def __init__(self):
        super().__init__()
        self.d_noise = 256
        self.d_classes = 200
        self.d_rest_code = 2
        self.d_code = self.d_classes + self.d_rest_code
        self.d_input = self.d_noise + self.d_code
        self.d_gen = 16

        self.define_module()

    def define_module(self):
        """
        Define each part of the network.
        """

        d_gen = self.d_gen
        self.fc_blocks = nn.Sequential(
            nn.Linear(self.d_input, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 16*d_gen*8*8),
            nn.BatchNorm1d(16*d_gen*8*8),
            nn.ReLU(inplace=True)
        )
        self.conv_blocks = nn.Sequential(
            nn.ConvTranspose2d(16*d_gen, 8*d_gen, 4, 2, 1),
            nn.BatchNorm2d(8*d_gen),
            nn.ReLU(),
            nn.ConvTranspose2d(8*d_gen, 4*d_gen, 4, 2, 1),
            nn.BatchNorm2d(4*d_gen),
            nn.ReLU(),
            nn.ConvTranspose2d(4*d_gen, 2*d_gen, 4, 2, 1),
            nn.BatchNorm2d(2*d_gen),
            nn.ReLU(),
            nn.ConvTranspose2d(2*d_gen, d_gen, 4, 2, 1),
            nn.BatchNorm2d(d_gen),
            nn.ReLU(),
            nn.ConvTranspose2d(d_gen, 3, 4, 2, 1)
        )

    def prepare_labels(self, labels):
        """
        Convert labels from long tensor to one hot encoded float tensor.
        """

        d_batch = labels.size(0)
        return torch.zeros(d_batch, self.d_classes, device=labels.device).scatter_(1, labels.unsqueeze(1), 1)

    def forward(self, noise, labels, rest_code):
        """
        Run the network forward.
        """
        # pylint: disable=arguments-differ
        # The arguments will differ from the base class since nn.Module is an abstract class.

        labels = self.prepare_labels(labels)
        latent = torch.cat((noise, labels, rest_code), dim=-1)
        latent = self.fc_blocks(latent)
        latent = latent.view(-1, 16*self.d_gen, 8, 8)
        images = self.conv_blocks(latent)
        return images

    def sample_latent(self, num_samples, device=None):
        """
        Sample latent vectors.
        """

        noise = torch.randn(num_samples, self.d_noise, device=device)
        labels = torch.randint(low=0, high=self.d_classes, size=(num_samples,), device=device)
        rest_code = torch.rand(num_samples, self.d_rest_code, device=device) * 2 - 1 # sample from uniform distribution over -1 to 1.
        return noise, labels, rest_code
