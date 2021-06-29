import torch
import torch.nn as nn

from mmt.models.builder import LOSS


@LOSS.register_module()
class MultiLabelBCEWithLogitsLoss(nn.Module):
    def __init__(self, loss_weight=1):
        super(MultiLabelBCEWithLogitsLoss, self).__init__()
        self.loss_weight = loss_weight
        # self.apply_onehot = apply_onehot
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, preds, gt_labels):
        """
        Args:
            preds (totorch.Tensor): (N, 82)
            gt_labels (list[torch.Tensor]): (NUM_C, )
        """
        gt_onehot_list = []
        for gt_label in gt_labels:
            gt_onehot = torch.zeros_like(preds[0])
            gt_onehot[gt_label] = 1
            gt_onehot_list.append(gt_onehot)
        gt_onehot = torch.cat([x[None] for x in gt_onehot_list], 0)
        return self.loss_weight * self.loss(preds, gt_onehot)

@LOSS.register_module()
class MultiLabelBCEWithLogitsFocalLoss(MultiLabelBCEWithLogitsLoss):
    def __init__(self, *args, gamma, **kwargs):
        super(MultiLabelBCEWithLogitsFocalLoss, self).__init__(*args, **kwargs)
        self.gamma = gamma
        self.loss = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, preds, gt_labels):
        ce_loss = super(MultiLabelBCEWithLogitsFocalLoss,
                        self).forward(preds, gt_labels) / self.loss_weight
        p = torch.exp(-ce_loss)
        loss = (1 - p) ** self.gamma * ce_loss
        return self.loss_weight * loss.mean()


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