"""
Tests for dcgan.
"""

import torch

from imgtxtgen.arch1.models.dcgan import DCGAN64, DCGAN256

def test_dcgan_64():
    """
    Test DCGAN for 64 x 64 images.
    """
    # pylint: disable=too-many-locals
    # This number of local variables is necessary for testing.

    gpu_id = 1
    d_batch = 16
    d_noise = 100
    d_gen = 16
    d_dis = 16

    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')

    model = DCGAN64(d_noise=d_noise, d_gen=d_gen, d_dis=d_dis).to(device)

    s_noise = (d_batch, d_noise, 1, 1)
    noise = torch.randn(s_noise, device=device)
    assert noise.size() == s_noise, 'noise size is incorrect.'

    pred_logits = model(noise)

    s_pred_logits = (d_batch, 1, 1, 1)
    assert pred_logits.size() == s_pred_logits, 'prediction logits size is incorrect.'

def test_dcgan_256():
    """
    Test DCGAN for 256 x 256 images.
    """
    # pylint: disable=too-many-locals
    # This number of local variables is necessary for testing.

    gpu_id = 1
    d_batch = 16
    d_noise = 100
    d_gen = 16
    d_dis = 16

    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')

    model = DCGAN256(d_noise=d_noise, d_gen=d_gen, d_dis=d_dis).to(device)

    s_noise = (d_batch, d_noise, 1, 1)
    noise = torch.randn(s_noise, device=device)
    assert noise.size() == s_noise, 'noise size is incorrect.'

    pred_logits = model(noise)

    s_pred_logits = (d_batch, 1, 1, 1)
    assert pred_logits.size() == s_pred_logits, 'prediction logits size is incorrect.'
