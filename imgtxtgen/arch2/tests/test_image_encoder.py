"""
Tests for ImageEncoder
"""

import torch

from imgtxtgen.arch2.models.image_encoder import ImageEncoder

def test_image_encoder():
    """
    Run tests for ImageEncoder.
    """

    d_batch = 20
    d_embed = 256

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = ImageEncoder(d_embed=d_embed).to(device)
    model.train()

    s_images = (d_batch, *model.input_shape)
    images = torch.randn(s_images, device=device)
    features = model(images)

    assert images.size() == s_images
    assert features.size() == (d_batch, d_embed)
