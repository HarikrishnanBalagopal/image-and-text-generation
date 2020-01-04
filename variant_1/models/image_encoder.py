"""
Module for an image encoder based on the inception v3 model.
We use the initial layers of the model which were pretrained on the ImageNet dataset.
The model produces both local (17 x 17 grid) and global image features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

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
        self.d_regions_rows = 17
        self.d_regions_col = 17

        self.input_shape = (self.d_ch, self.d_height, self.d_width)
        self.output_shape = ((self.d_embed, self.d_regions_rows, self.d_regions_col), (self.d_embed,))

        self.define_module()

    def define_module(self):
        """
        Define each part of the image encoder.
        Most layers are taken from the inception v3 model.
        """

        incv3_model = models.inception_v3(pretrained=True)
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

        self.fc_local_features = nn.Conv2d(768, self.d_embed, kernel_size=1, stride=1, padding=0, bias=False)
        self.fc_global_features = nn.Linear(2048, self.d_embed)

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

        # local image region features
        local_image_features = x
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
        # 2048

        # global image features
        global_image_features = self.fc_global_features(x)
        # d_batch x d_embed
        local_image_features = self.fc_local_features(local_image_features)
        # d_batch x d_embed x 17 x 17

        return local_image_features, global_image_features

def run_tests():
    """
    Run tests for ImageEncoder.
    """
    # pylint: disable=invalid-name
    # Short variable names like x, h and c are fine in this context.

    d_batch = 20
    d_embed = 256
    lr = 0.01

    print('running tests for ImageEncoder:')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device:', device)

    # test construction
    print('testing construction:')
    model = ImageEncoder(d_embed=d_embed).to(device)
    print('model:', model)
    print('input shape:', model.input_shape, 'output shape:', model.output_shape)

    # test training
    print('testing training:')
    # forward pass
    print('testing forward pass:')
    model.train()
    xs = torch.randn(d_batch, *model.input_shape).to(device)
    print('xs:', xs.size())
    ys = model(xs)
    print('ys:', len(ys), 'ys local:', ys[0].size(), 'ys global:', ys[1].size())
    assert xs.size()[1:] == model.input_shape, f'input shape is incorrect, expected:{model.input_shape} actual:{xs.size()[1:]}'
    assert ys[0].size()[1:] == model.output_shape[0], f'output 0 shape is incorrect, expected:{model.output_shape[0]} actual:{ys[0].size()[1:]}'
    assert ys[1].size()[1:] == model.output_shape[1], f'output 1 shape is incorrect, expected:{model.output_shape[1]} actual:{ys[1].size()[1:]}'

    # backward pass
    print('testing backward pass:')
    target_0 = torch.randn_like(ys[0])
    target_1 = torch.randn_like(ys[1])
    loss_fn = nn.MSELoss()
    loss_0 = loss_fn(ys[0], target_0)
    loss_1 = loss_fn(ys[1], target_1)
    loss = loss_0 + loss_1
    print('loss:', loss)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

if __name__ == '__main__':
    run_tests()
