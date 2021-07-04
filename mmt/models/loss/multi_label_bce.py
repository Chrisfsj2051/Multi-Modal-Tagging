import torch
import torch.nn as nn

from mmt.models.builder import LOSS
import numpy as np
import torch.nn.functional as F


@LOSS.register_module()
class MultiLabelBCEWithLogitsLoss(nn.Module):
    def __init__(self, loss_weight=1):
        super(MultiLabelBCEWithLogitsLoss, self).__init__()
        self.loss_weight = loss_weight
        # self.apply_onehot = apply_onehot
        self.loss = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, preds, gt_labels, ignore_labels=None):
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
        loss = self.loss(preds, gt_onehot)
        if ignore_labels is None:
            return self.loss_weight * loss.mean()
        return self.loss_weight * 1


@LOSS.register_module()
class MixupMultiLabelBCEWithLogitsLoss(nn.Module):
    def __init__(self, loss_weight=1):
        super(MixupMultiLabelBCEWithLogitsLoss, self).__init__()
        self.loss_weight = loss_weight
        # self.apply_onehot = apply_onehot
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, preds, gt_labels, gt_alpha):
        """
        Args:
            preds (totorch.Tensor): (N, 82)
            gt_labels (list[torch.Tensor]): (NUM_C, )
        """
        soft_gt_labels = []
        for i, gt_label in enumerate(gt_labels):
            gt_onehot_alpha = torch.zeros_like(preds[0])
            gt_onehot_alpha[gt_label] = gt_alpha[i]
            soft_gt_labels.append(gt_onehot_alpha)

        soft_gt_labels = torch.cat([x[None] for x in soft_gt_labels], 0)
        return self.loss_weight * self.loss(preds, soft_gt_labels)


@LOSS.register_module()
class MultiLabelBCEWithLogitsFocalLoss(MultiLabelBCEWithLogitsLoss):
    def __init__(self, *args, gamma, use_alpha=False, **kwargs):
        super(MultiLabelBCEWithLogitsFocalLoss, self).__init__(*args, **kwargs)
        self.gamma = gamma
        self.use_alpha = use_alpha
        self.loss = nn.BCEWithLogitsLoss(reduction='none')
        self.alpha = [0.6830567800156528, 0.7734705126218591, 0.8407812195093809, 0.8664331007507253,
                      0.9221381815312124, 0.9368252320701604, 0.9544844962796698, 0.9683316606413728,
                      0.9747262687805766, 0.9811069235182559, 0.9871390199860243, 0.987808493739435, 0.990317675009207,
                      0.9899832400402204, 0.992156395045811, 0.9941609769103517, 0.9926576669858066, 0.9948288714390605,
                      0.9956635294781647, 0.9964979543529732, 0.9951627626552717, 0.9963310880155284,
                      0.9971653265931645, 0.1434756298252353, 0.6131388291084047, 0.5930172422532473,
                      0.6471601102053066, 0.6909898524075501, 0.7528615031121385, 0.8966780481055574,
                      0.9025481087005501, 0.9267575769292556, 0.9434603546382736, 0.9488929594924852,
                      0.9529606169277136, 0.9644543965094109, 0.9566842249341152, 0.9654663568926982,
                      0.9722037567088072, 0.9742219405980315, 0.9812746484392536, 0.9809391890389209,
                      0.9861345254666692, 0.9842920656387756, 0.9886451233148115, 0.988310499814843, 0.9889797090660617,
                      0.990317675009207, 0.035244643529416295, 0.2214769829654946, 0.301309987254234,
                      0.48696030502959387, 0.7270639573672549, 0.7563397738241755, 0.7570713615910637,
                      0.7751040305436951, 0.8413109196367962, 0.8495951182123278, 0.8553956195711023,
                      0.8795121860115352, 0.8723703828693263, 0.8788162776203591, 0.894949122857697, 0.899614674350059,
                      0.910985599241904, 0.9140785422818559, 0.918711429363139, 0.9419305564867746, 0.9593892712676939,
                      0.9663093871962027, 0.96006513551716, 0.9706892019637816, 0.9713624345835137, 0.9775824879959086,
                      0.9827837430518577, 0.9898160084331585, 0.992323495070547, 0.9926576669858066, 0.9964979543529732,
                      0.9959973273895678, 0.9958304330958483, 0.9973321463887984, ]
        self.alpha = np.array(self.alpha)

    def forward(self, preds, gt_labels):
        ce_loss = super(MultiLabelBCEWithLogitsFocalLoss,
                        self).forward(preds, gt_labels) / self.loss_weight
        p = torch.exp(-ce_loss)
        loss = (1 - p) ** self.gamma * ce_loss
        if self.use_alpha:
            loss = loss * loss.new_tensor(self.alpha[None])
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
