"""
Tests for ImageEncoder.
"""

import torch

from torch.nn import MSELoss
from torch.optim import Adam
from imgtxtgen.arch4.models.image_encoder import ImageEncoder

def test_image_encoder():
    """
    Run the tests for ImageEncoder.
    Trains on d_image_size x d_image_size color images for num_iterations iterations.
    Checks that training loss is lower at the end of training.
    """

    d_batch = 20
    d_embed = 256
    d_image_size = 1024
    num_iterations = 10
    s_features = (d_batch, d_embed)
    s_images = (d_batch, 3, d_image_size, d_image_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    images = torch.randn(s_images, device=device, requires_grad=False)
    target = torch.randn(s_features, device=device)
    assert images.size() == s_images
    assert target.size() == s_features

    criterion = MSELoss()

    model = ImageEncoder(d_embed).to(device)
    model.train()
    assert model

    optimizer = Adam(model.parameters())

    with torch.no_grad():
        features = model(images)
        initial_loss = criterion(features, target).item()
    assert features.size() == s_features

    for _ in range(num_iterations):
        features = model(images)
        loss = criterion(features, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        features = model(images)
        final_loss = criterion(features, target).item()

    assert final_loss < initial_loss
