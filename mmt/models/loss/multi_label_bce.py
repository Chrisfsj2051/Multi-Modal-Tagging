import torch
import torch.nn as nn

from mmt.models.builder import LOSS


@LOSS.register_module()
class MultiLabelBCEWithLogitsLoss(nn.Module):
    def __init__(self, loss_weight=1):
        super(MultiLabelBCEWithLogitsLoss, self).__init__()
        self.loss_weight = loss_weight

    def forward(self, preds, gt_labels):
        """
        Args:
            preds (torch.Tensor): (82, )
            gt_labels (torch.Tensor): (NUM_C, )
        """
        gt_onehot = torch.zeros_like(preds)
        gt_onehot[gt_labels] = 1
        return self.loss_weight * nn.BCEWithLogitsLoss()(preds, gt_onehot)
