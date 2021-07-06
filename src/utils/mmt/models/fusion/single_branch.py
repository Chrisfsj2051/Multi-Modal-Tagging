import torch
from mmcv import print_log

from src.utils.mmt.models import ARCH, build_backbone, build_head
from src.utils.mmt.models import BaseFusionModel
from src.utils.mmt.utils import get_root_logger


@ARCH.register_module()
class SingleBranchModel(BaseFusionModel):
    def __init__(self, key, backbone, head, pretrained=None):
        super(SingleBranchModel, self).__init__()
        self.backbone = build_backbone(backbone)
        self.head = build_head(head)
        self.key = key
        if pretrained is not None:
            self.backbone.init_weights(pretrained=pretrained)

    def forward_train(self, return_feats=False, gt_labels_ignore=None, **kwargs):
        assert len(kwargs) == 3  # x, meta, label
        args = list(kwargs.values())
        x, meta_info, gt_labels = args
        feats = self.backbone(x, meta_info)
        losses = self.head.forward_train(feats, meta_info, gt_labels, gt_labels_ignore)
        if not return_feats:
            return losses
        else:
            return feats, losses

    def simple_test(self, return_feats=False, **kwargs):
        assert len(kwargs) == 2  # x, meta
        args = list(kwargs.values())
        x, meta_info = args
        feats = self.backbone(x, meta_info)
        preds = self.head.simple_test(feats, meta_info)
        if not return_feats:
            return [{self.key: preds}]
        else:
            return feats, [{self.key: preds}]


@ARCH.register_module()
class SemiSingleBranchModel(SingleBranchModel):

    def __init__(self, gt_thr, unlabeled_loss_weight=1.0, use_ema=True, **kwargs):
        super(SemiSingleBranchModel, self).__init__(**kwargs)
        self.burnin = True
        self.gt_thr = gt_thr
        self.use_ema = use_ema
        self.unlabeled_loss_weight = unlabeled_loss_weight

    def unlabeled_forward_train(self, **kwargs):
        self.eval()
        if self.use_ema:
            self.ema_hook._swap_ema_parameters()
        with torch.no_grad():
            pseudo_labels = self.simple_test(return_feats=False, **kwargs['weak'])[0].values()
        assert len(pseudo_labels) == 1
        pseudo_mask = list(pseudo_labels)[0].sigmoid()
        pseudo_labels = []
        for idx in range(pseudo_mask.shape[0]):
            pseudo_label = (pseudo_mask[idx] >= self.gt_thr).nonzero(as_tuple=False)
            pseudo_label = pseudo_label.view(pseudo_label.shape[0])
            pseudo_labels.append(pseudo_label)
        if self.use_ema:
            self.ema_hook._swap_ema_parameters()
        self.train()
        kwargs['strong']['gt_labels'] = pseudo_labels
        for i, item in enumerate(pseudo_labels):
            if item.numel() == 0:
                print_log('Empty Pseudo Label, skip', logger=get_root_logger())
                return None
            # label_list = [self.index_to_tag[x.item()] for x in item]
            # print(kwargs['strong']['meta_info'][i]['id_name'], label_list)

        return super(SemiSingleBranchModel, self).forward_train(return_feats=False, **kwargs['strong'])

    def forward_train(self, return_feats=False, **kwargs):
        extra_data = kwargs.pop('extra')
        losses = super(SemiSingleBranchModel,
                       self).forward_train(return_feats=return_feats, **kwargs)
        assert not return_feats
        # if return_feats:
        #     feats, losses = losses
        if not self.burnin:
            unlabeled_loss = self.unlabeled_forward_train(**extra_data)
            if unlabeled_loss is None:
                for key in list(losses.keys()):
                    losses[f'ssl_{key}'] = torch.zeros_like(losses[key])
            # if return_feats:
            #     feats, losses = losses
            else:
                for key in unlabeled_loss.keys():
                    losses[f'ssl_{key}'] = unlabeled_loss[key] * self.unlabeled_loss_weight
        if not return_feats:
            return losses
        else:
            return feats, losses

    def simple_test(self, return_feats=False, **kwargs):
        assert len(kwargs) == 2  # x, meta
        args = list(kwargs.values())
        x, meta_info = args
        feats = self.backbone(x, meta_info)
        preds = self.head.simple_test(feats, meta_info)
        if not return_feats:
            return [{self.key: preds}]
        else:
            return feats, [{self.key: preds}]
