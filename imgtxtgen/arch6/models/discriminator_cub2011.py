"""
CUB2011 version.
Discriminator for InfoGAN.
Architecture taken from the InfoGAN paper:
https://arxiv.org/pdf/1606.03657.pdf
Architecture code:
https://github.com/openai/InfoGAN/blob/master/infogan/models/regularized_gan.py
"""

from torch import nn

def _conv_block(d_in, d_out, use_bn=True):
    """
    Returns layers of each discriminator block
    """

    block = [nn.Conv2d(d_in, d_out, 4, 2, 1)]
    if use_bn:
        block.append(nn.BatchNorm2d(d_out)) # 0.8?
    block.append(nn.LeakyReLU(0.1, inplace=True))
    return block

class Discriminator(nn.Module):
    """
    Discriminator of InfoGAN for CUB2011.
    Contains both D and Q networks.
    Built for MNIST dataset: 28 x 28 grayscale images from 10 classes.
    """

    def __init__(self):
        super().__init__()
        self.d_ch = 3
        self.d_dis = 16

        self.define_module()

    def define_module(self):
        """
        Define each part of the network.
        """

        d_dis = self.d_dis
        self.common_blocks = nn.Sequential(
            *_conv_block(self.d_ch, d_dis, use_bn=False),
            *_conv_block(d_dis, 2*d_dis),
            *_conv_block(2*d_dis, 4*d_dis),
            *_conv_block(4*d_dis, 8*d_dis),
            *_conv_block(8*d_dis, 16*d_dis),
            nn.Flatten(),
            nn.Linear(16*d_dis*8*8, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.1)
        )
    def forward(self, images):
        """
        Run the network forward.
        """
        # pylint: disable=arguments-differ
        # The arguments will differ from the base class since nn.Module is an abstract class.

        return self.common_blocks(images)

class DHead(nn.Module):
    """
    Discriminator head for predicting whether images are real or fake.
    """

    def __init__(self):
        super().__init__()
        self.define_module()

    def define_module(self):
        """
        Define each part of D_head.
        """

        self.fc_d = nn.Linear(1024, 1)

    def forward(self, features):
        """
        Run the network forward.
        """
        # pylint: disable=arguments-differ
        # The arguments will differ from the base class since nn.Module is an abstract class.

        return self.fc_d(features)

class QHead(nn.Module):
    """
    Discriminator head for predicting the latent code.
    """

    def __init__(self):
        super().__init__()
        self.d_classes = 200
        self.d_rest_code = 2
        self.d_code = self.d_classes + 2*self.d_rest_code # 2*d_rest_code since we predict mean and variance of the normal distribution.

        self.define_module()

    def define_module(self):
        """
        Define each part of Q_head.
        """
        self.fc_q = nn.Sequential(
            nn.Linear(1024, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, self.d_code)
        )

    def forward(self, features):
        """
        Run the network forward.
        """
        # pylint: disable=arguments-differ
        # pylint: disable=bad-whitespace
        # The arguments will differ from the base class since nn.Module is an abstract class.
        # The whitespace makes it more readable.

        d_classes    = self.d_classes
        d_rest_code  = self.d_rest_code
        code         = self.fc_q(features)
        label_logits = code[:, :d_classes]
        rest_means   = code[:, d_classes:d_classes+d_rest_code]
        rest_vars    = code[:, d_classes+d_rest_code:].exp()

        return label_logits, rest_means, rest_vars
