"""
Deep Convolutional Generative Adversarial Network (DCGAN) is a generative model
that can be sampled to get images. The network uses 2d convolutional layers.
"""

import torch
import torch.nn as nn

class DCGAN(nn.Module):
    """
    The DCGAN network.
    """

    def __init__(self):
        super().__init__()

    def forward(self):
        # pylint: disable=arguments-differ
        # pylint: disable=invalid-name
        # The arguments will differ from the base class since nn.Module is an abstract class.
        # Short variable names like x, h and c are fine in this context.
        pass
