import torch.nn as nn

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
        out  = self.linear(x)
        if self.use_dropout:
            out = self.dropout(out)
        return out


@HEAD.register_module()
class ClsHead(FCHead):
    def __init__(self, in_dim, out_dim, loss):
        super(ClsHead, self).__init__(in_dim, out_dim)
        self.loss = build_loss(loss)

    def forward_train(self, x, gt_labels):
        pred = self.linear(x)
        return [
            self.loss(pred[i], gt_labels[i]) for i in range(len(x))
        ]

    def simple_test(self, x):
        return self.linear(x)
