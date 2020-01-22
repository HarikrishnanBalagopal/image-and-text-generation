"""
Tests for InfoGAN discriminator.
"""

import torch

from imgtxtgen.arch6.models.discriminator import Discriminator

def test_discriminator():
    """
    Test the dicriminator.
    """
    d_batch = 64
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Discriminator()
    model.train().to(device)
    images = torch.randn(d_batch, 1, 28, 28, device=device)
    valid_logits, label_logits, rest_means = model(images)

    assert valid_logits.size() == (d_batch, 1)
    assert label_logits.size() == (d_batch, 10)
    assert rest_means.size() == (d_batch, 2)
