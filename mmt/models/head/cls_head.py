import torch.nn as nn

from mmt.models.builder import HEAD, build_loss


@HEAD.register_module()
class ClsHead(nn.Module):
    def __init__(self, loss):
        super(ClsHead, self).__init__()
        self.linear = nn.Linear(16384, 82)
        self.loss = build_loss(loss)

    def forward_train(self, x, gt_labels):
        pred = self.linear(x)
        losses = {}
        losses['cls_loss'] = [
            self.loss(pred[i], gt_labels[i]) for i in range(len(x))
        ]
        return losses

    def simple_test(self, x):
        return self.linear(x)
