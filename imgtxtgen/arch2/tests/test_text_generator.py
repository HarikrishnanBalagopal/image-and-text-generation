"""
Tests for TextGenerator.
"""

import torch

from imgtxtgen.arch2.models.text_generator import TextGenerator

def test_image_encoder():
    """
    Run tests for TextGenerator.
    """

    d_ch = 3
    d_batch = 20
    d_image_size = 256

    opts = {'d_vocab':1000, 'd_embed':256, 'd_hidden':768, 'd_max_seq_len':18, 'd_image_features':2048, 'd_noise': 256}

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = TextGenerator(**opts).to(device)
    model.train()

    s_images = (d_batch, d_ch, d_image_size, d_image_size)
    s_noise = (d_batch, opts['d_noise'])
    images = torch.randn(s_images, device=device)
    noise = torch.randn(s_noise, device=device)
    captions, captions_log_probs, hiddens = model(images, noise)

    assert images.size() == s_images
    assert captions.size() == (d_batch, opts['d_max_seq_len'])
    assert captions_log_probs.size() == (d_batch, opts['d_vocab'], opts['d_max_seq_len'])
    assert hiddens[0].size() == (d_batch, opts['d_hidden'], opts['d_max_seq_len'])
    assert hiddens[1].size() == (d_batch, opts['d_hidden'], opts['d_max_seq_len'])
