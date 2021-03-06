"""
Tests for RecursiveLSTM.
"""

import torch

from torch.optim import Adam
from torch.nn import NLLLoss
from imgtxtgen.arch2.models.recursive_lstm import RecursiveLSTM

def test_recursive_lstm():
    """
    Run tests for RecursiveLSTM.
    Trains on a small dataset using MLE.
    """
    # pylint: disable=too-many-locals
    # This number of local variables is necessary since this test is also training the model.

    d_batch = 4
    num_iterations = 10
    opts = {'d_vocab':1000, 'd_embed':256, 'd_hidden':768, 'd_max_seq_len':18, 'd_condition':2048}
    s_image_features = (d_batch, opts['d_condition'])
    image_features = torch.randn(s_image_features)
    targets = [
        [2, 3, 4, 5, 6, 7, 8],
        [2, 4, 6, 8, 10, 12, 14],
        [3, 6, 3, 6, 3, 6, 3, 6],
        [2, 2, 2, 4, 3, 3]
    ]

    model = RecursiveLSTM(**opts)

    targets = model.prepare_targets(targets)
    assert targets.size() == (d_batch, opts['d_max_seq_len'])

    loss_fn = NLLLoss()
    optimizer = Adam(model.parameters(), lr=0.001)

    for _ in range(1, num_iterations+1):
        captions, captions_log_probs, hiddens = model(image_features)

        assert captions.size() == (d_batch, opts['d_max_seq_len'])
        assert captions_log_probs.size() == (d_batch, opts['d_vocab'], opts['d_max_seq_len'])
        assert hiddens[0].size() == (d_batch, opts['d_hidden'], opts['d_max_seq_len'])
        assert hiddens[1].size() == (d_batch, opts['d_hidden'], opts['d_max_seq_len'])

        loss = loss_fn(captions_log_probs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
