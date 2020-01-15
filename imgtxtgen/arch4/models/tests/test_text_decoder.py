"""
Tests for TextDecoder.

def debug():
    # DEBUG START-----------------------------------
    # captions <class 'torch.Tensor'> torch.Size([4, 18])
    # embeddings 1 <class 'torch.Tensor'> torch.Size([4, 18, 256])
    # embeddings 2 <class 'torch.Tensor'> torch.Size([4, 19, 256])
    # packed <class 'torch.nn.utils.rnn.PackedSequence'> torch.Size([43, 256]) tensor([4, 4, 4, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1])
    # hiddens <class 'torch.nn.utils.rnn.PackedSequence'> torch.Size([43, 256]) tensor([4, 4, 4, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1])
    # hiddens[0] <class 'torch.Tensor'> torch.Size([43, 256])
    # outputs <class 'torch.Tensor'> torch.Size([43, 3450])
    # DEBUG END-----------------------------------
    # ys: <class 'torch.Tensor'> torch.Size([43, 3450])

    # 43 = 18 + 14 + 8 + 3

    td = TextDecoder(256, 256, 3450, 1, 18)
    fs = torch.randn(4, 256)
    cs = torch.randint(low=2, high=3450, size=(4, 18))
    ls = torch.LongTensor([18, 14, 8, 3])
    ys = td(fs, cs, ls)
    print('ys:', type(ys), ys.size())
"""

import torch

from imgtxtgen.arch4.models.text_decoder import TextDecoder

def test_image_encoder():
    """
    Run the tests for TextDecoder.
    """

    d_batch = 4
    d_embed = 256
    d_hidden = 256
    d_vocab = 3450
    d_layers = 1
    d_max_seq_len = 18

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    features = torch.randn(d_batch, d_embed, device=device)
    captions = torch.randint(low=2, high=d_vocab, size=(d_batch, d_max_seq_len), device=device)
    lengths = torch.tensor([18, 14, 8, 3], dtype=torch.long, device=device)

    num_words = lengths.sum().item()

    model = TextDecoder(d_embed, d_hidden, d_vocab, d_layers, d_max_seq_len=d_max_seq_len).to(device)
    model.train()
    assert model

    pred_logits = model(features, captions, lengths)
    assert pred_logits.size() == (num_words, d_vocab)
