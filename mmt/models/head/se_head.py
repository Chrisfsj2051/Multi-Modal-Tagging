import torch
import torch.nn as nn
from torch.nn import init

from mmt.models.head import ClsHead
from mmt.models.builder import HEAD, build_loss


"""
TODO:
1. SE-GATING
2. LAYER NORM
3. TWO STAGE TRAINING
4. ...
"""


@HEAD.register_module()
class SEHead(ClsHead):
    def __init__(self, in_dim, hidden_size, out_dim, gating_reduction, loss):
        super(SEHead, self).__init__(hidden_size, out_dim, loss)
        self.hidden_size = hidden_size
        self.in_dim = in_dim
        self.hidden_weight = nn.Parameter(torch.randn(in_dim, hidden_size))
        self.hidden_bn = nn.BatchNorm1d(hidden_size)
        self.gatting_weight_1 = nn.Parameter(torch.randn(hidden_size, hidden_size//gating_reduction))
        self.gatting_bn_1 = nn.BatchNorm1d(hidden_size // gating_reduction)
        self.gatting_weight_2 = nn.Parameter(torch.randn(hidden_size // gating_reduction, hidden_size))
        # self.gatting_bn_2 = nn.BatchNorm1d(hidden_size)
        for layer in (self.hidden_weight, self.gatting_weight_1, self.gatting_weight_2):
            init.kaiming_uniform_(layer, mode='fan_in')
    
    def forward(self, x):
        assert x.shape[1] == self.in_dim
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
        return super(SEHead, self).forward_train(activation, gt_labels)
    
    def simple_test(self, x):
        activation = self(x)
        return super(SEHead, self).simple_test(activation)
