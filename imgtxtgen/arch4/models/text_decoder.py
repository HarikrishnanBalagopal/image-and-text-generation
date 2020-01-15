"""
Network to decode image features as text.
Model architecture taken from pytorch image captioning tutorial.
https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/03-advanced/image_captioning/model.py
"""

import torch

from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

class TextDecoder(nn.Module):
    """
    Network to decode image features into text.
    """

    def __init__(self, d_embed, d_hidden, d_vocab, d_layers, d_max_seq_len=20):
        super().__init__()
        self.d_vocab = d_vocab
        self.d_embed = d_embed
        self.d_hidden = d_hidden
        self.d_layers = d_layers
        self.d_max_seq_len = d_max_seq_len

        self.define_module()

    def define_module(self):
        """
        Set the hyper-parameters and build the layers.
        """
        self.embed = nn.Embedding(self.d_vocab, self.d_embed)
        self.lstm = nn.LSTM(self.d_embed, self.d_hidden, self.d_layers, batch_first=True)
        self.linear = nn.Linear(self.d_hidden, self.d_vocab)

    def forward(self, features, captions, lengths):
        """
        Decode image feature vectors and generates captions.
        """
        # pylint: disable=arguments-differ
        # The arguments will differ from the base class since nn.Module is an abstract class.

        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), dim=1)
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True)
        hiddens, _ = self.lstm(packed)
        outputs = self.linear(hiddens[0])
        return outputs

    def sample(self, features, states=None):
        """
        Generate captions for given image features using greedy search.
        """
        sampled_ids = []
        inputs = features.unsqueeze(1)
        for _ in range(self.d_max_seq_len):
            hiddens, states = self.lstm(inputs, states) # hiddens: (batch_size, 1, hidden_size)
            outputs = self.linear(hiddens.squeeze(1))   # outputs:  (batch_size, vocab_size)
            _, predicted = outputs.max(1)               # predicted: (batch_size)
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)              # inputs: (batch_size, embed_size)
            inputs = inputs.unsqueeze(1)                # inputs: (batch_size, 1, embed_size)
        sampled_ids = torch.stack(sampled_ids, 1)       # sampled_ids: (batch_size, max_seq_length)
        return sampled_ids
