"""
Module for an image encoder based on the inception v3 model.
We use the initial layers of the model which were pretrained on the ImageNet dataset.
"""

import torch.nn.functional as F

from torch import nn
from torchvision.models import inception_v3

class ImageEncoder(nn.Module):
    """
    Image encoder based on inception v3.
    """
    # pylint: disable=too-many-instance-attributes
    # The attributes are necessary.

    def __init__(self, d_embed):
        super().__init__()
        self.d_embed = d_embed
        self.d_width = 299
        self.d_height = 299
        self.d_ch = 3
        self.use_fc = self.d_embed != 2048

        self.input_shape = (self.d_ch, self.d_height, self.d_width)
        self.output_shape = (self.d_embed,)

        self.define_module()

    def define_module(self):
        """
        Define each part of the image encoder.
        Most layers are taken from the inception v3 model.
        """

        incv3_model = inception_v3(pretrained=True)
        for param in incv3_model.parameters():
            param.requires_grad = False

        self.conv2d_1a_3x3 = incv3_model.Conv2d_1a_3x3
        self.conv2d_2a_3x3 = incv3_model.Conv2d_2a_3x3
        self.conv2d_2b_3x3 = incv3_model.Conv2d_2b_3x3
        self.conv2d_3b_1x1 = incv3_model.Conv2d_3b_1x1
        self.conv2d_4a_3x3 = incv3_model.Conv2d_4a_3x3
        self.mixed_5b = incv3_model.Mixed_5b
        self.mixed_5c = incv3_model.Mixed_5c
        self.mixed_5d = incv3_model.Mixed_5d
        self.mixed_6a = incv3_model.Mixed_6a
        self.mixed_6b = incv3_model.Mixed_6b
        self.mixed_6c = incv3_model.Mixed_6c
        self.mixed_6d = incv3_model.Mixed_6d
        self.mixed_6e = incv3_model.Mixed_6e
        self.mixed_7a = incv3_model.Mixed_7a
        self.mixed_7b = incv3_model.Mixed_7b
        self.mixed_7c = incv3_model.Mixed_7c

        if self.use_fc:
            self.fc_features = nn.Linear(2048, self.d_embed)

    def forward(self, x):
        """
        Run the image encoder on the input images to get the features.
        """
        # pylint: disable=arguments-differ
        # The arguments will differ from the base class since nn.Module is an abstract class.

        # --> fixed-size input: d_batch x d_ch x d_H x d_W
        x = F.interpolate(x, size=(self.d_height, self.d_width), mode='bilinear', align_corners=False)
        # 299 x 299 x 3
        x = self.conv2d_1a_3x3(x)
        # 149 x 149 x 32
        x = self.conv2d_2a_3x3(x)
        # 147 x 147 x 32
        x = self.conv2d_2b_3x3(x)
        # 147 x 147 x 64
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 73 x 73 x 64
        x = self.conv2d_3b_1x1(x)
        # 73 x 73 x 80
        x = self.conv2d_4a_3x3(x)
        # 71 x 71 x 192

        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 35 x 35 x 192
        x = self.mixed_5b(x)
        # 35 x 35 x 256
        x = self.mixed_5c(x)
        # 35 x 35 x 288
        x = self.mixed_5d(x)
        # 35 x 35 x 288

        x = self.mixed_6a(x)
        # 17 x 17 x 768
        x = self.mixed_6b(x)
        # 17 x 17 x 768
        x = self.mixed_6c(x)
        # 17 x 17 x 768
        x = self.mixed_6d(x)
        # 17 x 17 x 768
        x = self.mixed_6e(x)
        # 17 x 17 x 768

        x = self.mixed_7a(x)
        # 8 x 8 x 1280
        x = self.mixed_7b(x)
        # 8 x 8 x 2048
        x = self.mixed_7c(x)
        # 8 x 8 x 2048
        x = F.avg_pool2d(x, kernel_size=8)
        # 1 x 1 x 2048
        # x = F.dropout(x, training=self.training)
        # 1 x 1 x 2048

        x = x.squeeze()
        # d_batch x 2048

        if self.use_fc:
            x = self.fc_features(x)
        # d_batch x d_embed

        return x
