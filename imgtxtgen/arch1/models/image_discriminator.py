"""
Module for the generator part of the DCGAN.
"""

import torch
import torch.nn as nn

def _weights_init(layer):
    """
    Custom weights initialization called on netG and netD.
    """

    classname = layer.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(layer.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(layer.weight.data, 1.0, 0.02)
        nn.init.constant_(layer.bias.data, 0)

class ImageDiscriminator256(nn.Module):
    """
    The discriminator part of the DCGAN for 256 x 256 images.
    """

    def __init__(self, d_dis, d_ch=3):
        super().__init__()
        self.d_ch = d_ch
        self.d_dis = d_dis
        self.define_module()
        self.model.apply(_weights_init)

    def define_module(self):
        """
        Define each part of the discriminator.
        """

        d_dis = self.d_dis
        self.model = nn.Sequential(
            # input is (nc) x 256 x 256
            nn.Conv2d(self.d_ch, d_dis, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 128 x 128
            nn.Conv2d(d_dis, d_dis * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(d_dis * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 64 x 64
            nn.Conv2d(d_dis * 2, d_dis * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(d_dis * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(d_dis * 4, d_dis * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(d_dis * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(d_dis * 8, d_dis * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(d_dis * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(d_dis * 16, d_dis * 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(d_dis * 32),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(d_dis * 32, 1, 4, 1, 0, bias=False),
            # state size. 1 x 1 x 1
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Run the model on the input.
        """
        # pylint: disable=arguments-differ
        # The arguments will differ from the base class since nn.Module is an abstract class.

        return self.model(x)

class ImageDiscriminator64(nn.Module):
    """
    The discriminator part of the DCGAN for 64 x 64 images.
    """

    def __init__(self, d_dis, d_ch=3):
        super().__init__()
        self.d_ch = d_ch
        self.d_dis = d_dis
        self.define_module()
        self.model.apply(_weights_init)

    def define_module(self):
        """
        Define each part of the discriminator.
        """

        d_dis = self.d_dis
        self.model = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(self.d_ch, d_dis, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(d_dis, d_dis * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(d_dis * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(d_dis * 2, d_dis * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(d_dis * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(d_dis * 4, d_dis * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(d_dis * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(d_dis * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Run the model on the input.
        """
        # pylint: disable=arguments-differ
        # The arguments will differ from the base class since nn.Module is an abstract class.

        return self.model(x)

def test_image_discriminator_256():
    """
    Test image discriminator for 256 x 256 images.
    """
    # pylint: disable=invalid-name
    # small variable names like x and y are fine for this test.

    print('Testing ImageDiscriminator256:')
    d_batch = 20
    d_dis = 64

    model = ImageDiscriminator256(d_dis=d_dis)
    print(model)

    s_images = (d_batch, 3, 256, 256)
    images = torch.randn(s_images)
    y = model(images)
    print('images:', images.size(), 'y:', y.size())

def test_image_discriminator_64():
    """
    Test image discriminator for 64 x 64 images.
    """
    # pylint: disable=invalid-name
    # small variable names like x and y are fine for this test.

    print('Testing ImageDiscriminator64:')
    d_batch = 20
    d_dis = 64

    model = ImageDiscriminator64(d_dis=d_dis)
    print(model)

    s_images = (d_batch, 3, 64, 64)
    images = torch.randn(s_images)
    y = model(images)
    print('images:', images.size(), 'y:', y.size())

def run_tests():
    """
    Run tests for generator.
    """
    test_image_discriminator_64()
    test_image_discriminator_256()

if __name__ == '__main__':
    run_tests()
