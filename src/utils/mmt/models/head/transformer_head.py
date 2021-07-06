from src.utils.mmt.models import HEAD, build_head
from src.utils.mmt.models import EncoderLayer
import torch.nn as nn
import torch


@HEAD.register_module()
class TransformerHead(nn.Module):
    def __init__(self, in_dim, num_head, num_layers, hidden_dim,
                 cls_head_config, dropout_p, transformer_hidden_dim=None):
        super(TransformerHead, self).__init__()
        if transformer_hidden_dim is None:
            transformer_hidden_dim = hidden_dim
        self.modal_fc = nn.ModuleDict()
        for key in in_dim.keys():
            self.modal_fc[key] = nn.Sequential(
                nn.Linear(in_dim[key], hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU()
            )
            # self.modal_fc[key] = SingleSEHead(
            #     in_dim=in_dim[key],
            #     out_dim=hidden_dim,
            #     gating_reduction=8,
            #     cls_head_config=None,
            #     norm_cfg=dict(type='BN1d'),
            #     dropout_p=dropout_p)
        self.encoder = nn.ModuleList([
            EncoderLayer(hidden_dim, num_head, transformer_hidden_dim)
            for _ in range(num_layers)
        ])
        self.cls_head = build_head(cls_head_config)
        if dropout_p == 0:
            dropout_p = 1e-11
        self.input_dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        # x = [self.modal_fc[key](self.input_dropout(val))[:, None, :] for key, val in x.items()]
        x = [self.modal_fc[key](val)[:, None, :] for key, val in x.items()]
        x = torch.cat(x, 1)
        for encoder in self.encoder:
            x = encoder(x)
        x = x.view(x.shape[0], -1)
        return x

    def forward_train(self, x, meta_info, gt_labels, gt_labels_ignore):
        activation = self(x)
        return self.cls_head.forward_train(activation, gt_labels, gt_labels_ignore)

    def simple_test(self, x, meta_info):
        activation = self(x)
        return self.cls_head.simple_test(activation)

# class SingleSEHead(nn.Module):
#     def __init__(self,
#                  in_dim,
#                  out_dim,
#                  gating_reduction,
#                  cls_head_config,
#                  norm_cfg=dict(type='BN1d'),
#                  dropout_p=0.0):
#         super(SingleSEHead, self).__init__()
#         self.cls_head = build_head(cls_head_config)
#         self.out_dim = out_dim
#         self.in_dim = in_dim
#         self.hidden_weight = nn.Parameter(torch.randn(in_dim, out_dim))
#         self.hidden_bn = build_norm_layer(norm_cfg, out_dim)[1]
#         self.gatting_weight_1 = nn.Parameter(
#             torch.randn(out_dim, out_dim // gating_reduction))
#         self.gatting_bn_1 = build_norm_layer(norm_cfg, out_dim // gating_reduction)[1]
#         self.gatting_weight_2 = nn.Parameter(
#             torch.randn(out_dim // gating_reduction, out_dim))
#         if dropout_p == 0:
#             dropout_p = 1e-11
#         self.input_dropout = nn.Dropout(dropout_p)
#         # self.gatting_bn_2 = nn.BatchNorm1d(out_dim)
#         for layer in (self.hidden_weight, self.gatting_weight_1,
#                       self.gatting_weight_2):
#             init.kaiming_uniform_(layer, mode='fan_in')
#
#     def forward(self, x):
#         assert x.shape[
#                    1] == self.in_dim, \
#             f'Input shape: {x.shape[1]}, Param shape: {self.in_dim}'
#         x = self.input_dropout(x)
#         activation = torch.matmul(x, self.hidden_weight)
#         activation = self.hidden_bn(activation)
#         gates = torch.matmul(activation, self.gatting_weight_1)
#         gates = nn.ReLU()(self.gatting_bn_1(gates))
#         gates = torch.matmul(gates, self.gatting_weight_2)
#         gates = gates.sigmoid()
#         activation = activation * gates
#         return activation
#
#     def forward_train(self, x, meta_info, gt_labels):
#         activation = self(x)
#         return self.cls_head.forward_train(activation, gt_labels)
#
#     def simple_test(self, x, meta_info):
#         activation = self(x)
#         return self.cls_head.simple_test(activation)
