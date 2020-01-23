"""
CUB2011 version.
Tests for InfoGAN.
"""

import torch

from imgtxtgen.arch6.models.infogan_cub2011 import InfoGAN

def test_infogan_cub2011():
    """
    Run test for InfoGAN.
    """

    d_batch = 64

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = InfoGAN().train().to(device)

    noise, labels, rest_code = model.gen.sample_latent(num_samples=d_batch, device=device)
    valid_logits, label_logits, rest_means, rest_vars = model(noise, labels, rest_code)

    assert valid_logits.size() == (d_batch, 1)
    assert label_logits.size() == (d_batch, 200)
    assert rest_means.size() == (d_batch, 2)
    assert rest_vars.size() == (d_batch, 2)
