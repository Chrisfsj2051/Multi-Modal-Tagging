import torch
import torch.nn as nn

from mmt.models.builder import LOSS


@LOSS.register_module()
class MultiLabelBCEWithLogitsLoss(nn.Module):
    def __init__(self, loss_weight=1, apply_onehot=True, with_sigmoid=True):
        super(MultiLabelBCEWithLogitsLoss, self).__init__()
        self.loss_weight = loss_weight
        self.apply_onehot = apply_onehot
        self.loss = (nn.BCEWithLogitsLoss() if with_sigmoid else nn.BCELoss())

    def forward(self, preds, gt_labels):
        """
        Args:
            preds (torch.Tensor): (82, )
            gt_labels (torch.Tensor): (NUM_C, )
        """
        if self.apply_onehot:
            gt_onehot = torch.zeros_like(preds)
            gt_onehot[gt_labels] = 1
        else:
            gt_onehot = gt_labels
            assert gt_onehot.shape == preds.shape
        return self.loss_weight * self.loss(preds, gt_onehot)

@LOSS.register_module()
class BCEWithLogitsLoss(nn.Module):
    def __init__(self, loss_weight=1):
        super(BCEWithLogitsLoss, self).__init__()
        self.loss = nn.BCEWithLogitsLoss()
        self.loss_weight = loss_weight

    def forward(self, preds, gt_labels):
        """
        Args:
            preds (torch.Tensor): (82, )
            gt_labels (torch.Tensor): (NUM_C, )
        """

        return self.loss_weight * self.loss(preds, gt_labels.float())