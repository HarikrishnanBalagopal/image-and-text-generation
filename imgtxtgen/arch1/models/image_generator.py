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

class ImageGenerator256(nn.Module):
    """
    The generator part of the DCGAN for 256 x 256 images.
    """
    def __init__(self, d_noise, d_gen, d_ch=3):
        super().__init__()
        self.d_noise = d_noise
        self.d_gen = d_gen
        self.d_ch = d_ch

        self.define_module()
        self.model.apply(_weights_init)

    def define_module(self):
        """
        Define each part of the generator.
        """

        d_gen = self.d_gen
        self.model = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(self.d_noise, d_gen * 32, 4, 1, 0, bias=False),
            nn.BatchNorm2d(d_gen * 32),
            nn.ReLU(True),
            # state size. (ngf*32) x 4 x 4
            nn.ConvTranspose2d(d_gen * 32, d_gen * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(d_gen * 16),
            nn.ReLU(True),
            # state size. (ngf*16) x 8 x 8
            nn.ConvTranspose2d(d_gen * 16, d_gen * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(d_gen * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 16 x 16
            nn.ConvTranspose2d(d_gen * 8, d_gen * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(d_gen * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 32 x 32
            nn.ConvTranspose2d(d_gen * 4, d_gen * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(d_gen * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 64 x 64
            nn.ConvTranspose2d(d_gen * 2, d_gen, 4, 2, 1, bias=False),
            nn.BatchNorm2d(d_gen),
            nn.ReLU(True),
            # state size. (ngf) x 128 x 128
            nn.ConvTranspose2d(d_gen, self.d_ch, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 256 x 256
        )

    def forward(self, x):
        """
        Run the model on the input.
        """
        # pylint: disable=arguments-differ
        # The arguments will differ from the base class since nn.Module is an abstract class.

        return self.model(x)

class ImageGenerator64(nn.Module):
    """
    The generator part of the DCGAN for 64 x 64 images.
    """
    def __init__(self, d_noise, d_gen, d_ch=3):
        super().__init__()
        self.d_noise = d_noise
        self.d_gen = d_gen
        self.d_ch = d_ch

        self.define_module()
        self.model.apply(_weights_init)

    def define_module(self):
        """
        Define each part of the generator.
        """

        d_gen = self.d_gen
        self.model = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(self.d_noise, d_gen * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(d_gen * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(d_gen * 8, d_gen * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(d_gen * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(d_gen * 4, d_gen * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(d_gen * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(d_gen * 2, d_gen, 4, 2, 1, bias=False),
            nn.BatchNorm2d(d_gen),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(d_gen, self.d_ch, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, x):
        """
        Run the model on the input.
        """
        # pylint: disable=arguments-differ
        # The arguments will differ from the base class since nn.Module is an abstract class.

        return self.model(x)

def test_image_generator_256():
    """
    Test ImageGenerator for 256 x 256 images.
    """
    # pylint: disable=invalid-name
    # small variable names like x and y are fine for this test.

    print('Testing ImageGenerator256:')
    d_batch = 20
    d_noise = 100
    d_gen = 64
    model = ImageGenerator256(d_noise=d_noise, d_gen=d_gen)
    print(model)
    s_noise = (d_batch, d_noise, 1, 1)
    noise = torch.randn(s_noise)
    images = model(noise)
    print('noise:', noise.size(), 'images:', images.size())

def test_image_generator_64():
    """
    Test ImageGenerator for 64 x 64 images.
    """
    # pylint: disable=invalid-name
    # small variable names like x and y are fine for this test.

    print('Testing ImageGenerator64:')
    d_batch = 20
    d_noise = 100
    d_gen = 64
    model = ImageGenerator64(d_noise=d_noise, d_gen=d_gen)
    print(model)
    s_noise = (d_batch, d_noise, 1, 1)
    noise = torch.randn(s_noise)
    images = model(noise)
    print('noise:', noise.size(), 'images:', images.size())

def run_tests():
    """
    Run tests for generator.
    """

    test_image_generator_64()
    test_image_generator_256()

if __name__ == '__main__':
    run_tests()
