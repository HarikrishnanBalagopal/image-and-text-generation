"""
Different image encoders for image captioning.
"""

from torch import nn
from torchvision.models import resnet152

class ImageEncoder1(nn.Module):
    """
    Takes in images and returns global image features. Based on resnet.
    input_shape = (d_batch, 3, d_image_size, d_image_size)
    output_shape = (d_batch, 2048)
    """

    def __init__(self):
        super().__init__()
        self.define_module()

    def define_module(self):
        resnet = resnet152(pretrained=True)  # pretrained ImageNet ResNet-152
        modules = list(resnet.children())[:-1] # Remove last linear layer.
        self.resnet = nn.Sequential(*modules, nn.Flatten()) # Flatten the 4D tensor to 2D.

    def forward(self, images):
        """
        Forward propagation.
        :param images: images, a tensor of dimensions (d_batch, 3, d_image_size, d_image_size)
        :return: encoded images (d_batch, 2048)
        """
        # pylint: disable=arguments-differ
        # The arguments will differ from the base class since nn.Module is an abstract class.

        return self.resnet(images)
