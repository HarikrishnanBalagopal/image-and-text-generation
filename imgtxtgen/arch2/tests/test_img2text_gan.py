"""
Tests for Img2TextGAN.
"""

from imgtxtgen.arch2.models.img2text_gan import Img2TextGAN

def test_img2text_gan():
    """
    Run tests for Img2TextGAN.
    """
    model = Img2TextGAN(d_vocab=256, d_embed=256, d_hidden=256, d_max_seq_len=18, d_image_features=256, d_noise=256, end_token=0, start_token=1)
    assert model
