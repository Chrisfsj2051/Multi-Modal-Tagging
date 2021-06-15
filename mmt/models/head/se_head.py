import torch
import torch.nn as nn
from mmcv.cnn import build_norm_layer
from torch.nn import init

from mmt.models.builder import HEAD, build_head

"""
TODO:
1. SE-GATING
2. LAYER NORM
3. TWO STAGE TRAINING
4. ...
"""


@HEAD.register_module()
class SingleSEHead(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 gating_reduction,
                 cls_head_config,
                 norm_cfg=dict(type='BN1d'),
                 dropout_p=0.0):
        super(SingleSEHead, self).__init__()
        self.cls_head = build_head(cls_head_config)
        self.out_dim = out_dim
        self.in_dim = in_dim
        self.hidden_weight = nn.Parameter(torch.randn(in_dim, out_dim))
        # self.hidden_bn = nn.BatchNorm1d(out_dim)
        self.hidden_bn = build_norm_layer(norm_cfg, out_dim)[1]
        # norm_cfg =
        # self.hidden_bn = build_norm_layer(norm_cfg, planes * block.expansion)[1]nn.BatchNorm1d(out_dim)
        self.gatting_weight_1 = nn.Parameter(
            torch.randn(out_dim, out_dim // gating_reduction))
        self.gatting_bn_1 = nn.BatchNorm1d(out_dim // gating_reduction)
        self.gatting_weight_2 = nn.Parameter(
            torch.randn(out_dim // gating_reduction, out_dim))
        if dropout_p == 0:
            dropout_p = 1e-11
        self.input_dropout = nn.Dropout(dropout_p)
        # self.gatting_bn_2 = nn.BatchNorm1d(out_dim)
        for layer in (self.hidden_weight, self.gatting_weight_1,
                      self.gatting_weight_2):
            init.kaiming_uniform_(layer, mode='fan_in')

    def forward(self, x):
        assert x.shape[1] == self.in_dim, f'Input shape: {x.shape[1]}, Param shape: {self.in_dim}'
        x = self.input_dropout(x)
        activation = torch.matmul(x, self.hidden_weight)
        activation = self.hidden_bn(activation)
        gates = torch.matmul(activation, self.gatting_weight_1)
        gates = nn.ReLU()(self.gatting_bn_1(gates))
        gates = torch.matmul(gates, self.gatting_weight_2)
        gates = gates.sigmoid()
        activation = activation * gates
        return activation

    def forward_train(self, x, gt_labels):
        activation = self(x)
        return self.cls_head.forward_train(activation, gt_labels)

    def simple_test(self, x):
        activation = self(x)
        return self.cls_head.simple_test(activation)

@HEAD.register_module()
class FusionSEHead(SingleSEHead):

    def forward_train(self, modal_inputs, feats_dict, gt_labels):
        x = torch.cat(list(feats_dict.values()), 1)
        return super(FusionSEHead, self).forward_train(x, gt_labels)

    def simple_test(self, modal_inputs, feats_dict):
        x = torch.cat(list(feats_dict.values()), 1)
        return super(FusionSEHead, self).simple_test(x)