from mmt.models.builder import ARCH, build_backbone, build_head
from mmt.models.fusion import BaseFusionModel


@ARCH.register_module()
class SingleBranchModel(BaseFusionModel):
    def __init__(self, key, backbone, head, pretrained=None):
        super(SingleBranchModel, self).__init__()
        self.backbone = build_backbone(backbone)
        self.head = build_head(head)
        self.key = key
        if pretrained is not None:
            self.backbone.init_weights(pretrained=pretrained)

    def forward_train(self, return_feats=False, **kwargs):
        assert len(kwargs) == 3  # x, meta, label
        args = list(kwargs.values())
        x, meta_info, gt_labels = args
        feats = self.backbone(x, meta_info)
        losses = self.head.forward_train(feats, meta_info, gt_labels)
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

    def __init__(self, **kwargs):
        super(SemiSingleBranchModel, self).__init__(**kwargs)
        # self.burn_in = True
        self.burn_in = False

    def unlabeled_forward_train(self, extra_data):
        print('in')

    def forward_train(self, return_feats=False, **kwargs):
        extra_data = kwargs.pop('extra')
        losses = super(SemiSingleBranchModel,
                             self).forward_train(return_feats=return_feats, **kwargs)
        assert not return_feats
        # if return_feats:
        #     feats, losses = losses
        if not self.burn_in:
            unlabeled_loss = self.unlabeled_forward_train(extra_data)
            # if return_feats:
            #     feats, losses = losses
            for key in unlabeled_loss.keys():
                losses[f'ssl_{key}'] = unlabeled_loss.pop(key)
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