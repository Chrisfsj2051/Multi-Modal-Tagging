import torch
import torch.nn as nn
from torch.nn import init

from mmt.models.builder import HEAD
"""
TODO:
1. SE-GATING
2. LAYER NORM
3. TWO STAGE TRAINING
4. ...
"""


@HEAD.register_module()
class SEHead(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 gating_reduction,
                 input_dropout_p=None):
        super(SEHead, self).__init__()
        self.out_dim = out_dim
        self.in_dim = in_dim
        self.hidden_weight = nn.Parameter(torch.randn(in_dim, out_dim))
        self.hidden_bn = nn.BatchNorm1d(out_dim)
        self.gatting_weight_1 = nn.Parameter(
            torch.randn(out_dim, out_dim // gating_reduction))
        self.gatting_bn_1 = nn.BatchNorm1d(out_dim // gating_reduction)
        self.gatting_weight_2 = nn.Parameter(
            torch.randn(out_dim // gating_reduction, out_dim))
        self.use_input_dropout = input_dropout_p is not None
        if self.use_input_dropout:
            self.input_dropout = nn.Dropout(p=input_dropout_p)
        # self.gatting_bn_2 = nn.BatchNorm1d(out_dim)
        for layer in (self.hidden_weight, self.gatting_weight_1,
                      self.gatting_weight_2):
            init.kaiming_uniform_(layer, mode='fan_in')

    def forward(self, x):
        assert x.shape[1] == self.in_dim
        if self.use_input_dropout:
            x = self.input_dropout(x)
        activation = torch.matmul(x, self.hidden_weight)
        activation = self.hidden_bn(activation)
        gates = torch.matmul(activation, self.gatting_weight_1)
        gates = self.gatting_bn_1(gates)
        gates = torch.matmul(gates, self.gatting_weight_2)
        gates = gates.sigmoid()
        activation = activation * gates
        return activation

    def forward_train(self, x, gt_labels):
        activation = self(x)
        return activation

    def simple_test(self, x):
        activation = self(x)
        return activation
