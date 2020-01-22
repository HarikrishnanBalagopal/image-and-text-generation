"""
Tests for InfoGAN generator.
"""

import torch

from imgtxtgen.arch6.models.generator import Generator

def test_generator():
    """
    Test the generator.
    """
    d_batch = 64
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Generator()
    model.train().to(device)
    noise, labels, rest_code = model.sample_latent(num_samples=d_batch, device=device)
    images = model(noise, labels, rest_code)

    assert noise.size() == (d_batch, 62)
    assert labels.size() == (d_batch,)
    assert rest_code.size() == (d_batch, 2)
    assert images.size() == (d_batch, 1, 28, 28)
