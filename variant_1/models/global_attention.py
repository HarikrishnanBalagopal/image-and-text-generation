"""
Global attention takes an annotations matrix and a query.
It computes a context vector as answer to the query.
The context vector is a convex combination of the annotations.
"""

import torch
import torch.nn as nn

class GlobalAttention(nn.Module):
    """
    GlobalAttention implemented according to the original paper:
    Learning to Align and Translate. https://arxiv.org/pdf/1409.0473.pdf
    """
    def __init__(self, d_query=0, d_annotations=0):
        super().__init__()
        self.d_query = d_query
        self.d_annotations = d_annotations
        self.use_fc_layer = d_query != d_annotations
        self.define_module()

    def define_module(self):
        """
        Define each part of the GlobalAttention model.
        """

        self.softmax = nn.Softmax(dim=2)
        if self.use_fc_layer:
            self.fc_layer = nn.Linear(self.d_query, self.d_annotations)

    def forward(self, queries, annotations):
        """
            queries    : d_batch x d_query
            annotations: d_batch x d_annotations x num_annotations. PyTorch convention is channels first.
        """
        # pylint: disable=arguments-differ
        # pylint: disable=invalid-name

        if self.use_fc_layer:
            queries = self.fc_layer(queries)
        attn_weights = self.softmax(torch.bmm(queries.unsqueeze(1), annotations))
        # d_batch x 1 x num_annotations
        contexts = torch.bmm(attn_weights, annotations.transpose(1, 2))
        # d_batch x 1 x d_annotations
        return contexts.squeeze(), attn_weights.squeeze()

def run_tests():
    """
    Run tests for global attention model.
    """

    print('query and annotations are of same dimension:')
    d_batch = 20
    d_query = 256
    d_annotations = d_query
    num_annotations = 17 * 17
    model = GlobalAttention(d_query=d_query, d_annotations=d_annotations)
    print(model, 'Using fc layer:', model.use_fc_layer)

    queries = torch.randn(d_batch, d_query)
    annotations = torch.randn(d_batch, d_annotations, num_annotations)
    contexts, attn_weights = model(queries, annotations)
    print('contexts:', contexts.size(), 'attn_weights:', attn_weights.size())

    print('query and annotations are of different dimensions:')
    d_batch = 20
    d_query = 256
    d_annotations = 768
    num_annotations = 17 * 17
    model = GlobalAttention(d_query=d_query, d_annotations=d_annotations)
    print(model, 'Using fc layer:', model.use_fc_layer)

    queries = torch.randn(d_batch, d_query)
    annotations = torch.randn(d_batch, d_annotations, num_annotations)
    contexts, attn_weights = model(queries, annotations)
    print('contexts:', contexts.size(), 'attn_weights:', attn_weights.size())

if __name__ == '__main__':
    run_tests()
