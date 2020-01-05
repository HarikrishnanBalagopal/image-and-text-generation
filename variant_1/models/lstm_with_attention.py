"""
Module for a RNN model that uses global attention (similar to show attend tell paper).
Uses a LSTM to generate the caption based on image features.
The image features must contain both local and global features.
"""

import torch
import torch.nn as nn
from global_attention import GlobalAttention

def _arg_max(logits):
    return torch.max(logits, dim=1)[1]

class LSTMWithAttention(nn.Module):
    """
    LSTMWithAttention is a LSTM that runs on it's own output at each time step.
    It also takes image features and uses global attention for generating the caption.
    """
    # pylint: disable=too-many-instance-attributes
    # The attributes are necessary.

    def __init__(self, d_vocab, d_embed, d_annotations, d_hidden, d_max_seq_len, d_global_image_features):
        # pylint: disable=too-many-arguments
        # The arguments are necessary.

        super().__init__()
        self.d_vocab = d_vocab
        self.d_embed = d_embed
        self.d_annotations = d_annotations
        self.d_input = self.d_embed + self.d_annotations
        self.d_hidden = d_hidden
        self.d_max_seq_len = d_max_seq_len
        self.d_global_image_features = d_global_image_features
        self.end_token = 0
        self.start_token = 1
        self.define_module()

    def define_module(self):
        """
        Define each part of the LSTMWithAttention module.
        """

        self.fc_h = nn.Linear(self.d_global_image_features, self.d_hidden)
        self.fc_c = nn.Linear(self.d_global_image_features, self.d_hidden)
        self.embed = nn.Embedding(num_embeddings=self.d_vocab, embedding_dim=self.d_embed)
        self.lstm_cell = nn.LSTMCell(input_size=self.d_input, hidden_size=self.d_hidden)
        self.fc_logits = nn.Linear(self.d_hidden, self.d_vocab)
        self.attn = GlobalAttention(d_query=self.d_hidden, d_annotations=self.d_annotations)

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

    def forward(self, image_features):
        """
        Run the LSTM forward, feeding the previous output as input using attention at each step.
        local_image_features : d_batch x d_annotations x 17 x 17
        global_image_features: d_batch x d_annotations
        """
        # pylint: disable=arguments-differ
        # pylint: disable=invalid-name
        # The arguments will differ from the base class since nn.Module is an abstract class.
        # Short variable names like x, h and c are fine in this context.

        local_image_features, global_image_features = image_features
        d_batch = local_image_features.size(0)
        local_image_features = local_image_features.view(d_batch, self.d_annotations, -1)

        x = torch.full((d_batch,), self.start_token, dtype=torch.long)
        captions = torch.full((d_batch, self.d_max_seq_len), self.end_token, dtype=torch.long)
        captions_logits = torch.empty(d_batch, self.d_vocab, self.d_max_seq_len)
        attn_maps = torch.empty(d_batch, self.d_max_seq_len, 17, 17)

        h, c = self.fc_h(global_image_features), self.fc_c(global_image_features)

        for t in range(self.d_max_seq_len):
            contexts, attn_weights = self.attn(h, local_image_features)
            attn_maps[:, t] = attn_weights.view(d_batch, 17, 17)

            x = self.embed(x)
            x = torch.cat((x, contexts), dim=1)
            h, c = self.lstm_cell(x, (h, c))
            x = self.fc_logits(h)
            captions_logits[:, :, t] = x
            x = _arg_max(x)
            captions[:, t] = x

        return captions, captions_logits, attn_maps

def run_tests():
    """
    Run tests for LSTMWithAttention.
    Trains the network on a simple dataset.
    """
    # pylint: disable=too-many-locals
    # This number of local variables is necessary since this test is also training the model.

    d_batch = 4
    num_iterations = 100
    opts = {'d_vocab':100, 'd_embed':256, 'd_annotations':768, 'd_hidden':768, 'd_max_seq_len':18, 'd_global_image_features':2048}
    s_local_image_features, s_global_image_features = (d_batch, opts['d_annotations'], 17, 17), (d_batch, opts['d_global_image_features'])
    local_image_features, global_image_features = torch.randn(*s_local_image_features), torch.randn(*s_global_image_features)
    image_features = (local_image_features, global_image_features)
    targets = [
        [2, 3, 4, 5, 6, 7, 8],
        [2, 4, 6, 8, 10, 12, 14],
        [3, 6, 3, 6, 3, 6, 3, 6],
        [2, 2, 2, 4, 3, 3]
    ]

    model = LSTMWithAttention(**opts)
    print(model)

    targets = model.prepare_targets(targets)
    print('prepared targets:', targets, targets.size())
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for i in range(1, num_iterations+1):
        print('iteration:', i)
        captions, captions_logits, attn_maps = model(image_features)
        loss = loss_fn(captions_logits, targets)
        print('captions:', captions)
        print('captions_logits:', captions_logits.size())
        print('attn_maps:', attn_maps.size())

        print('loss:', loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

if __name__ == '__main__':
    run_tests()
