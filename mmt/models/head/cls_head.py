import numpy as np
import torch
import torch.nn as nn

from mmt import TaggingDataset
from mmt.models.builder import HEAD, build_loss


@HEAD.register_module()
class FCHead(nn.Module):
    def __init__(self, in_dim, out_dim, dropout_p=None):
        super(FCHead, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.use_dropout = dropout_p is not None
        if self.use_dropout:
            self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        out = self.linear(x)
        if self.use_dropout:
            out = self.dropout(out)
        return out


@HEAD.register_module()
class ClsHead(FCHead):
    def __init__(self, in_dim, out_dim, loss, dropout_p=None):
        super(ClsHead, self).__init__(in_dim, out_dim, dropout_p)
        self.loss = build_loss(loss)

    def forward_train(self, x, gt_labels):
        if self.use_dropout:
            x = self.dropout(x)
        pred = self.linear(x)
        loss_list = [self.loss(pred[i], gt_labels[i]) for i in range(len(x))]
        return dict(cls_loss=loss_list)

    def simple_test(self, x):
        return self.linear(x)


@HEAD.register_module()
class HMCHead(nn.Module):
    def __init__(self,
                 in_dim,
                 feat_dim,
                 out_dim,
                 loss,
                 label_id_file,
                 dropout_p=None):
        super(HMCHead, self).__init__()
        self.loss = build_loss(loss)
        (_, _, self.index_to_super_index,
         _) = TaggingDataset.load_label_dict(label_id_file)
        num_super_class = np.unique(list(self.index_to_super_index.values()))
        num_super_class = len(num_super_class)
        self.num_super_class = num_super_class
        wg_list, wt_list, wl_list = [], [], []
        for i in range(num_super_class):
            prev_dim = in_dim if i == 0 else in_dim + feat_dim
            wg_list.append(
                nn.Sequential(nn.Linear(prev_dim, feat_dim), nn.ReLU()))
            wt_list.append(
                nn.Sequential(nn.Linear(feat_dim, feat_dim), nn.ReLU()))
            wl_list.append(
                nn.Sequential(nn.Linear(feat_dim, out_dim), nn.Sigmoid()))
        wg_list.append(
            nn.Sequential(nn.Linear(feat_dim, out_dim), nn.Sigmoid()))
        self.wg_list = nn.ModuleList(wg_list)
        self.wl_list = nn.ModuleList(wl_list)
        self.wt_list = nn.ModuleList(wt_list)

    def forward(self, x, gt_labels=None, with_loss=False):
        feat_in = x
        local_preds = []
        for i in range(self.num_super_class):
            if i != 0:
                feat_in = torch.cat([x, feat_in], 1)
            feat_in = self.wg_list[i](feat_in)
            feat_A = self.wt_list[i](feat_in)
            local_preds.append(self.wl_list[i](feat_A))
        global_outputs = self.wg_list[-1](feat_in)
        local_combined = torch.zeros_like(global_outputs)
        if with_loss:
            losses = {}
            gt_onehot = torch.zeros_like(global_outputs)
            for i, gt_lab in enumerate(gt_labels):
                gt_onehot[i][gt_lab] = 1
        for i in range(self.num_super_class):
            mask = [
                int(self.index_to_super_index[x] == i)
                for x in self.index_to_super_index.keys()
            ]
            mask = torch.BoolTensor(mask).cuda()
            local_combined[:, mask] = local_preds[i][:, mask]
            if with_loss:
                losses[f'HMC_loc_loss_{i}'] = self.loss(
                    local_preds[i][:, mask], gt_onehot[:, mask])
        global_preds = global_outputs * 0.5 + local_combined * 0.5
        if with_loss:
            losses['HMC_global_loss'] = self.loss(global_preds, gt_onehot)
            return losses
        else:
            return global_preds

    def forward_train(self, x, gt_labels):
        return self(x, gt_labels, with_loss=True)

    def simple_test(self, x):
        return self(x)


@HEAD.register_module()
class MLPHead(ClsHead):
    def __init__(self, in_dim, out_dim, **kwargs):
        super(MLPHead, self).__init__(in_dim, out_dim, **kwargs)
        self.linear = nn.Sequential(nn.Linear(in_dim, 512),
                                    nn.BatchNorm1d(512), nn.ReLU(),
                                    nn.Dropout(0.5),
                                    nn.Linear(512, 256),
                                    nn.BatchNorm1d(256), nn.ReLU(),
                                    nn.Dropout(0.5),
                                    nn.Linear(256, out_dim),
                                    )

    def forward_train(self, x, gt_labels):
        if self.use_dropout:
            x = self.dropout(x)
        pred = self.linear(x)
        loss_list = [self.loss(pred[i], gt_labels[i]) for i in range(len(x))]
        return dict(cls_loss=loss_list)

    def simple_test(self, x):
        return self.linear(x)
