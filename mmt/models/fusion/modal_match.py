from mmt.models.fusion import BaseFusionModel
from mmt.models.builder import ARCH, build_head, build_backbone


@ARCH.register_module()
class ModalMatchModel(BaseFusionModel):
    def __init__(self, backbone_config, head_config,  modal_keys):
        assert len(modal_keys) == 2
        super(ModalMatchModel, self).__init__()
        self.backbone_a = build_backbone(backbone_config[modal_keys[0]])
        self.backbone_b = build_backbone(backbone_config[modal_keys[1]])
        self.head = build_head(head_config)
        self.modal_keys = modal_keys

    def forward_train(self, meta_info, gt_labels, **kwargs):
        modal_a_feats = self.backbone_a(kwargs[self.modal_keys[0]], meta_info)
        modal_b_feats = self.backbone_b(kwargs[self.modal_keys[1]], meta_info)
        return self.head.forward_train([modal_a_feats, modal_b_feats], meta_info, gt_labels)

    def simple_test(self, meta_info, **kwargs):
        modal_a_feats = self.backbone_a(kwargs[self.modal_keys[0]], meta_info)
        modal_b_feats = self.backbone_b(kwargs[self.modal_keys[1]], meta_info)
        return self.head.simple_test([modal_a_feats, modal_b_feats], meta_info)