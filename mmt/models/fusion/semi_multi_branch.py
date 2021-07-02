import random

import numpy as np
import torch
from mmcv import print_log

from mmt.models.fusion import MultiBranchFusionModel
from mmt.models.builder import ARCH
from mmt.utils import get_root_logger


@ARCH.register_module()
class SemiMultiBranchFusionModel(MultiBranchFusionModel):
    def __init__(self, unlabeled_loss_weight, gt_thr, **kwargs):
        super(SemiMultiBranchFusionModel, self).__init__(**kwargs)
        self.unlabeled_loss_weight = unlabeled_loss_weight
        self.gt_thr = gt_thr

    def forward_train(self, **kwargs):
        extra_data = kwargs.pop('extra')
        losses = super(SemiMultiBranchFusionModel, self).forward_train( **kwargs)
        if not self.burnin:
            unlabeled_loss = self.unlabeled_forward_train(**extra_data)
            if unlabeled_loss is None:
                for key in list(losses.keys()):
                    losses[f'ssl_{key}'] = torch.zeros_like(losses[key])
            else:
                for key in unlabeled_loss.keys():
                    losses[f'ssl_{key}'] = unlabeled_loss[key] * self.unlabeled_loss_weight
        return losses

    def unlabeled_forward_train(self, **kwargs):
        self.eval()
        self.ema_hook._swap_ema_parameters()
        with torch.no_grad():
            pseudo_labels = self.simple_test(**kwargs['weak'])[0]['fusion']
        pseudo_mask = pseudo_labels.sigmoid()
        pseudo_labels = []
        for idx in range(pseudo_mask.shape[0]):
            pseudo_label = (pseudo_mask[idx] >= self.gt_thr).nonzero(as_tuple=False)
            pseudo_label = pseudo_label.view(pseudo_label.shape[0])
            pseudo_labels.append(pseudo_label)
        self.ema_hook._swap_ema_parameters()
        self.train()
        kwargs['strong']['gt_labels'] = pseudo_labels
        for i, item in enumerate(pseudo_labels):
            if item.numel() == 0:
                print_log('Empty Pseudo Label, skip', logger=get_root_logger())
                return None

        return super(SemiMultiBranchFusionModel, self).forward_train(**kwargs['strong'])

