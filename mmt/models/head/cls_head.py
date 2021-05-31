import torch.nn as nn

from mmt.models.builder import HEAD, build_loss


@HEAD.register_module()
class FCHead(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(FCHead, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.linear(x)


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
