"""
Tests for TextDiscriminator.
"""

from imgtxtgen.arch2.models.text_discriminator import TextDiscriminator

def test_text_discriminator():
    """
    Run tests for TextDiscriminator.
    """

    d_vocab = 3450
    d_embed = 256
    d_hidden = 256

    model = TextDiscriminator(d_vocab=d_vocab, d_embed=d_embed, d_hidden=d_hidden)
    assert model


