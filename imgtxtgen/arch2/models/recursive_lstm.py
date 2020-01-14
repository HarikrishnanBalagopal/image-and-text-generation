"""
Module for a RNN model that uses its own output from previous timestep as input at current timestep.
Uses a LSTM to generate text based on a condition.
"""

import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

def _sample_from(logits):
    return Categorical(logits=logits).sample()

class RecursiveLSTM(nn.Module):
    """
    RecursiveLSTM is a LSTM that uses output at t - 1 as input at t.
    Generates text based on a condition.
    """
    # pylint: disable=too-many-instance-attributes
    # The attributes are necessary.

    def __init__(self, d_vocab, d_embed, d_hidden, d_max_seq_len, d_condition, end_token=0, start_token=1):
        # pylint: disable=too-many-arguments
        # The arguments are necessary.

        super().__init__()
        self.d_vocab = d_vocab
        self.d_embed = d_embed
        self.d_hidden = d_hidden
        self.end_token = end_token
        self.start_token = start_token
        self.d_condition = d_condition
        self.d_max_seq_len = d_max_seq_len

        self.define_module()

    def define_module(self):
        """
        Define each part of the LSTMWithAttention module.
        """

        self.fc_h = nn.Linear(self.d_condition, self.d_hidden)
        self.fc_c = nn.Linear(self.d_condition, self.d_hidden)
        self.embed = nn.Embedding(num_embeddings=self.d_vocab, embedding_dim=self.d_embed)
        self.lstm_cell = nn.LSTMCell(input_size=self.d_embed, hidden_size=self.d_hidden)
        self.fc_logits = nn.Linear(self.d_hidden, self.d_vocab)

    def prepare_targets(self, targets):
        """
        Slice, append end token, pad to correct length and convert to long tensor.
        targets: list of sequences.
        """

        s_target = (len(targets), self.d_max_seq_len)
        target_tensor = torch.full(s_target, self.end_token, dtype=torch.long)
        for i, target in enumerate(targets):
            target_tensor[i, :len(target)] = torch.LongTensor(target)
        return target_tensor

    def forward(self, condition):
        """
        Run the LSTM forward, feeding the previous output as input at each step.
        condition : d_batch x d_condition
        """
        # pylint: disable=arguments-differ
        # pylint: disable=invalid-name
        # pylint: disable=bad-whitespace
        # The arguments will differ from the base class since nn.Module is an abstract class.
        # Short variable names like x, h and c are fine in this context. The whitespace makes it more readable.

        d_batch              = condition.size(0)
        device               = condition.device
        x                    = torch.full((d_batch,)                   , self.start_token, dtype=torch.long, device=device)
        captions             = torch.full((d_batch, self.d_max_seq_len), self.end_token  , dtype=torch.long, device=device)
        captions_logits      = torch.empty((d_batch, self.d_vocab, self.d_max_seq_len)                     , device=device)

        h, c = self.fc_h(condition), self.fc_c(condition)

        for t in range(self.d_max_seq_len):
            x = self.embed(x)
            h, c = self.lstm_cell(x, (h, c))
            x = self.fc_logits(h)
            captions_logits[:, :, t] = x
            x = _sample_from(x)
            captions[:, t] = x

        return captions, captions_logits
