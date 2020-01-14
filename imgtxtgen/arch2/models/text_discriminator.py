"""
Network to learn the reward function for reinforcement learning.
"""

from torch import nn

class TextDiscriminator(nn.Module):
    """
    LSTM to learn reward function for reinforcement learning.
    """

    def __init__(self, d_vocab, d_embed, d_hidden):
        super().__init__()
        self.d_vocab = d_vocab
        self.d_embed = d_embed
        self.d_hidden = d_hidden

        self.define_module()

    def define_module(self):
        """
        Define each part of the TextDiscriminator.
        """
        self.embed = nn.Embedding(num_embeddings=self.d_vocab, embedding_dim=self.d_embed)
        self.rnn = nn.LSTM(input_size=self.d_embed, hidden_size=self.d_hidden, batch_first=True)
        self.fc_logits = nn.Linear(self.d_hidden, 1)

    def forward(self, x):
        """
        Run the network forward on input.
        x : (d_batch, d_max_seq_len)
        """
        # pylint: disable=arguments-differ
        # pylint: disable=invalid-name
        # pylint: disable=bad-whitespace
        # The arguments will differ from the base class since nn.Module is an abstract class.
        # Short variable names like x, h and c are fine in this context. The whitespace makes it more readable.

        x = self.embed(x)
        _, (h, _) = self.rnn(x)
        x = self.fc_logits(h)
        return x
