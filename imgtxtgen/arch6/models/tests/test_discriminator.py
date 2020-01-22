"""
Tests for InfoGAN discriminator.
"""

import torch

from imgtxtgen.arch6.models.discriminator import Discriminator, DHead, QHead

def test_discriminator_part_of_infogan():
    """
    Test the dicriminator.
    """
    d_batch = 64
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Discriminator().train().to(device)
    images = torch.randn(d_batch, 1, 28, 28, device=device)
    features = model(images)

    assert features.size() == (d_batch, 1024)

    d_head = DHead().train().to(device)
    q_head = QHead().train().to(device)

    valid_logits = d_head(features)

    assert valid_logits.size() == (d_batch, 1)

    label_logits, rest_means, rest_vars = q_head(features)

    assert label_logits.size() == (d_batch, 10)
    assert rest_means.size() == (d_batch, 2)
    assert rest_vars.size() == (d_batch, 2)
