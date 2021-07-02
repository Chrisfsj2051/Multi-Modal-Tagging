import torch
import torch.nn as nn
from mmcv.cnn import build_norm_layer
from torch.nn import init
import torch.nn as nn
from mmt.models.builder import HEAD, build_head
import numpy as np


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
        self.hidden_bn = build_norm_layer(norm_cfg, out_dim)[1]
        self.gatting_weight_1 = nn.Parameter(
            torch.randn(out_dim, out_dim // gating_reduction))
        self.gatting_bn_1 = build_norm_layer(norm_cfg, out_dim // gating_reduction)[1]
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
        assert x.shape[
                   1] == self.in_dim, \
            f'Input shape: {x.shape[1]}, Param shape: {self.in_dim}'
        x = self.input_dropout(x)
        activation = torch.matmul(x, self.hidden_weight)
        activation = self.hidden_bn(activation)
        gates = torch.matmul(activation, self.gatting_weight_1)
        gates = nn.ReLU()(self.gatting_bn_1(gates))
        gates = torch.matmul(gates, self.gatting_weight_2)
        gates = gates.sigmoid()
        activation = activation * gates
        return activation

    def forward_train(self, x, meta_info, gt_labels):
        activation = self(x)
        return self.cls_head.forward_train(activation, gt_labels)

    def simple_test(self, x, meta_info):
        activation = self(x)
        return self.cls_head.simple_test(activation)


@HEAD.register_module()
class FusionSEHead(SingleSEHead):
    def forward_train(self, feats_dict, meta_info, gt_labels):
        x = torch.cat(list(feats_dict.values()), 1)
        return super(FusionSEHead, self).forward_train(x, meta_info, gt_labels)

    def simple_test(self, feats_dict, meta_info):
        x = torch.cat(list(feats_dict.values()), 1)
        return super(FusionSEHead, self).simple_test(x, meta_info)


@HEAD.register_module()
class FusionSEHeadWithModalAttn(FusionSEHead):

    def __init__(self, *args, modal_in_dim, **kwargs):
        super(FusionSEHeadWithModalAttn, self).__init__(*args, **kwargs)
        self.attn_module = nn.ModuleDict()
        for key, val in modal_in_dim.items():
            self.attn_module[key] = nn.Sequential(nn.Linear(val, 1024), nn.BatchNorm1d(1024), nn.ReLU(),
                                                  nn.Linear(1024, 1), nn.Sigmoid())

    def forward_train(self, feats_dict, meta_info, gt_labels):
        for key in feats_dict.keys():
            # print(key, self.attn_module[key](feats_dict[key]))
            feats_dict[key] = feats_dict[key] * self.attn_module[key](feats_dict[key])
        return super(FusionSEHeadWithModalAttn, self).forward_train(feats_dict, meta_info, gt_labels)

    def simple_test(self, feats_dict, meta_info):
        for key in feats_dict.keys():
            feats_dict[key] = feats_dict[key] * self.attn_module[key](feats_dict[key])
        return super(FusionSEHeadWithModalAttn, self).simple_test(feats_dict, meta_info)


@HEAD.register_module()
class SingleMixupSEHead(SingleSEHead):

    def forward_train(self, x, meta_info, gt_labels):
        new_x, new_gt_labels, new_gt_alpha = [], [], []
        bs = len(gt_labels)
        for idx in range(bs):
            alpha = np.random.beta(1.5, 1.5)
            if alpha < 0.5:
                alpha = 1 - alpha
            nx_idx = (bs // 2 + idx) % bs
            x1, x2 = x[idx], x[nx_idx]
            tmp_gt_label = {i: 0.0 for i in range(max(
                gt_labels[idx].tolist() + gt_labels[nx_idx].tolist()) + 1)}
            for label in gt_labels[idx].tolist():
                tmp_gt_label[label] += alpha
            for label in gt_labels[nx_idx].tolist():
                tmp_gt_label[label] += 1 - alpha
            label = [k for (k, v) in tmp_gt_label.items() if v > 0]
            label_alpha = [v for (k, v) in tmp_gt_label.items() if v > 0]
            new_gt_labels.append(gt_labels[0].new_tensor(label))
            new_gt_alpha.append(x1.new_tensor(label_alpha))
            new_x.append((x1 * alpha + x2 * (1 - alpha))[None])

        new_x = torch.cat(new_x, 0)
        activation = self(new_x)
        return self.cls_head.forward_train(activation, new_gt_labels, new_gt_alpha)
