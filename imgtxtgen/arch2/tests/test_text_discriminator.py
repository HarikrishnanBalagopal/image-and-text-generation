"""
Tests for TextDiscriminator.
"""

import torch

from imgtxtgen.arch2.models.text_discriminator import TextDiscriminator

def test_text_discriminator():
    """
    Run tests for TextDiscriminator.
    """

    d_batch = 20
    d_vocab = 3450
    d_embed = 256
    d_hidden = 256
    d_max_seq_len = 18

    model = TextDiscriminator(d_vocab=d_vocab, d_embed=d_embed, d_hidden=d_hidden)
    assert model

    s_captions = (d_batch, d_max_seq_len)
    captions = torch.randint(low=2, high=d_vocab, size=s_captions)
    pred_logits = model(captions)

    assert captions.size() == s_captions
    assert pred_logits.size() == (d_batch,)
