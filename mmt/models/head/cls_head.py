import numpy as np
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
        return [self.loss(pred[i], gt_labels[i]) for i in range(len(x))]

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
        wg_list, wt_list, wl_list = [], [], []
        for i in range(num_super_class):
            prev_dim = in_dim if i == 0 else in_dim + feat_dim
            wg_list.append(
                nn.Sequential(nn.Linear(prev_dim, feat_dim), nn.ReLU()))
            wt_list.append(
                nn.Sequential(nn.Linear(feat_dim, feat_dim), nn.ReLU()))
            wl_list.append(
                nn.Sequential(nn.Linear(feat_dim, out_dim), nn.Sigmoid()))
        wg_list.append(nn.Sequential(nn.Linear(prev_dim, out_dim), nn.ReLU()))

    def forward_train(self, x, gt_labels):
        # if self.use_dropout:
        #     x = self.dropout(x)
        print('in')
        pred = self.linear(x)
        return [self.loss(pred[i], gt_labels[i]) for i in range(len(x))]

    def simple_test(self, x):
        return self.linear(x)


@HEAD.register_module()
class MLPHead(ClsHead):
    def __init__(self, in_dim, out_dim, **kwargs):
        super(MLPHead, self).__init__(in_dim, out_dim, **kwargs)
        self.linear = nn.Sequential(nn.Linear(in_dim, 512),
                                    nn.BatchNorm2d(512), nn.ReLU(),
                                    nn.Linear(512, out_dim))

    def forward_train(self, x, gt_labels):
        if self.use_dropout:
            x = self.dropout(x)
        pred = self.linear(x)
        return [self.loss(pred[i], gt_labels[i]) for i in range(len(x))]

    def simple_test(self, x):
        return self.linear(x)
