"""
Module for a RNN model that uses global attention (similar to show attend tell paper).
Uses a LSTM to generate the caption based on image features.
The image features must contain both local and global features.
"""

import torch
import torch.nn as nn
from global_attention import MyGlobalAttention

def _arg_max(logits):
    return torch.max(logits, dim=1)[1].item()

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
        self.start_token = 0
        self.end_token = 1
        self.define_module()

    def define_module(self):
        """
        Define each part of the LSTMWithAttention module.
        """

        self.embed = nn.Embedding(num_embeddings=self.d_vocab, embedding_dim=self.d_embed)
        self.lstm_cell = nn.LSTMCell(input_size=self.d_input, hidden_size=self.d_hidden)
        self.h_fc = nn.Linear(self.d_global_image_features, self.d_hidden)
        self.c_fc = nn.Linear(self.d_global_image_features, self.d_hidden)
        self.fc_logits = nn.Linear(self.d_hidden, self.d_vocab)
        self.attn = MyGlobalAttention()

    def prepare_target(self, target):
        """
        Slice, append end token, pad to correct length and convert to long tensor.
        """

        target = target[:self.d_max_seq_len - 1]
        num_padding = self.d_max_seq_len - len(target)
        target += [self.end_token] * num_padding
        return torch.LongTensor(target)

    def forward(self, image_features):
        """
        Run the LSTM forward, feeding the previous output as input.
        assume batch size = 1 and use attention at each step.
        local_image_features : d_batch x 768 x 17 x 17
        """
        # pylint: disable=arguments-differ
        # pylint: disable=invalid-name
        # The arguments will differ from the base class since nn.Module is an abstract class.
        # Short variable names like x, h and c are fine in this context.

        output = []
        attn_maps = []
        output_logits = []
        x = torch.LongTensor([self.start_token])
        local_image_features, global_image_features = image_features

        h, c = self.h_fc(global_image_features), self.c_fc(global_image_features)

        local_image_features = local_image_features.view(self.d_annotations, -1).t()

        for _ in range(self.d_max_seq_len):
            context, attn_weights = self.attn(h, local_image_features)
            attn_maps.append(attn_weights)

            x = self.embed(x)
            x = torch.cat((x, context), dim=1)
            h, c = self.lstm_cell(x, (h, c))
            x = self.fc_logits(h)
            output_logits.append(x)
            x = _arg_max(x)
            output.append(x)
            x = torch.LongTensor([x])

        return output, torch.cat(output_logits, dim=0), attn_maps

def run_tests():
    """
    Run tests for LSTMWithAttention.
    Trains the network on a simple dataset.
    """

    d_batch = 4
    num_iterations = 100
    opts = {
        'd_vocab':100, 'd_embed':256, 'd_annotations':768, 'd_hidden':768, 'd_max_seq_len':18, 'd_global_image_features':2048
    }
    targets = [
        [2, 3, 4, 5, 6, 7, 8],
        [2, 4, 6, 8, 10, 12, 14],
        [3, 6, 3, 6, 3, 6, 3, 6],
        [2, 2, 2, 4, 3, 3]
    ]
    local_image_features_shape, global_image_features_shape = (opts['d_annotations'], 17, 17), (opts['d_global_image_features'],)

    my_lstm = LSTMWithAttention(**opts)
    local_image_features, global_image_features = torch.randn(d_batch, *local_image_features_shape), torch.randn(d_batch, *global_image_features_shape)
    targets = [my_lstm.prepare_target(target) for target in targets]
    print('prepared target:', targets, targets[0].size())
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(my_lstm.parameters(), lr=0.001)

    for i in range(1, num_iterations+1):
        print('iteration:', i)
        for local_image_feature, global_image_feature, target in zip(local_image_features, global_image_features, targets):
            print('current target:', target)
            image_features = (local_image_feature.unsqueeze(0), global_image_feature.unsqueeze(0))
            pred, pred_logits, attn_maps = my_lstm(image_features)
            print('attn_maps:', len(attn_maps), attn_maps[0].size())
            print('predictions:', pred, len(pred_logits), pred_logits.size())
            loss = loss_fn(pred_logits, target)
            print('loss:', loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

if __name__ == '__main__':
    run_tests()
