"""
Global attention takes an annotations matrix and a query.
It computes a context vector as answer to the query.
The context vector is a convex combination of the annotations.
"""

import torch
import torch.nn as nn

class MyGlobalAttention(nn.Module):
    """
    MyGlobalAttention implemented according to the original paper:
    Learning to Align and Translate. https://arxiv.org/pdf/1409.0473.pdf
    """
    def __init__(self, use_fc_layer=False, d_query=0, d_annotations=0):
        super().__init__()
        self.use_fc_layer = use_fc_layer
        if self.use_fc_layer:
            self.d_query = d_query
            self.d_annotations = d_annotations
        self.define_module()

    def define_module(self):
        """
        define the parts of MyGlobalAttention.
        """

        self.get_weight_logits = lambda query, annotations: torch.matmul(query, annotations.t())
        self.softmax = nn.Softmax(dim=1)
        if self.use_fc_layer:
            self.fc_layer = nn.Linear(self.d_query, self.d_annotations)

    def forward(self, query, annotations):
        """
            query: d_batch x d_annotations
            annotations: num_annotations x d_annotations
        """
        # pylint: disable=arguments-differ
        # pylint: disable=invalid-name

        if self.use_fc_layer:
            query = self.fc_layer(query)
        attn_weights = self.softmax(self.get_weight_logits(query, annotations))
        context = torch.matmul(attn_weights, annotations)
        return context, attn_weights
