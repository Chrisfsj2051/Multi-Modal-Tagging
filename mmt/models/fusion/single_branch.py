from mmt.models.builder import ARCH, build_backbone, build_head
from mmt.models.fusion import BaseFusionModel


@ARCH.register_module()
class SingleBranchModel(BaseFusionModel):
    def __init__(self, backbone, head, pretrained=None):
        super(SingleBranchModel, self).__init__()
        self.backbone = build_backbone(backbone)
        self.head = build_head(head)
        if pretrained is not None:
            self.backbone.init_weights(pretrained=pretrained)

    def forward_train(self, **kwargs):
        assert len(kwargs) == 3  # x, meta, label
        args = list(kwargs.values())
        x, meta_info, gt_labels = args
        feats = self.backbone(x, meta_info)
        losses = self.head.forward_train(feats, meta_info, gt_labels)
        return losses

    def simple_test(self, **kwargs):
        assert len(kwargs) == 2  # x, meta
        args = list(kwargs.values())
        x, meta_info = args
        feats = self.backbone(x, meta_info)
        preds = self.head.simple_test(feats, meta_info)
        return preds
