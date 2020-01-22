"""
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
    Discriminator for InfoGAN.
    Contains both D and Q networks.
    Built for MNIST dataset: 28 x 28 grayscale images from 10 classes.
    """

    def __init__(self):
        super().__init__()
        self.d_ch = 1
        self.d_dis = 64
        self.d_classes = 10
        self.d_rest_code = 2
        self.d_code = self.d_classes + self.d_rest_code

        self.define_module()

    def define_module(self):
        """
        Define each part of the network.
        """

        d_dis = self.d_dis
        self.common_blocks = nn.Sequential(
            *_conv_block(self.d_ch, d_dis, use_bn=False),
            *_conv_block(d_dis, d_dis*2),
            nn.Flatten(),
            nn.Linear(d_dis*2*7*7, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.1)
        )
        self.fc_d = nn.Linear(1024, 1)
        self.fc_q = nn.Sequential(
            nn.Linear(1024, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, self.d_code)
        )


    def forward(self, images):
        """
        Run the network forward.
        """
        # pylint: disable=arguments-differ
        # pylint: disable=bad-whitespace
        # The arguments will differ from the base class since nn.Module is an abstract class.
        # The whitespace makes it more readable.

        latent       = self.common_blocks(images)
        valid_logits = self.fc_d(latent)
        code         = self.fc_q(latent)
        label_logits = code[:, :self.d_classes]
        rest_means   = code[:, self.d_classes:]

        return valid_logits, label_logits, rest_means
