from mmt.models import BaseFusionModel
from mmt.models.builder import ARCH, build_model, build_head, build_backbone


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
        modal_a_feats = self.branch_a(kwargs[self.modal_keys[0]])
        modal_b_feats = self.branch_b(kwargs[self.modal_keys[1]])
        return self.head.forward_train([modal_a_feats, modal_b_feats], meta_info, gt_labels)

    def simple_test(self, video, image, text, audio, meta_info):
        all_preds = {}
        video_feats, video_preds = self.video_branch.simple_test(
            video=video, meta_info=meta_info, return_feats=True)
        image_feats, image_preds = self.image_branch.simple_test(
            image=image, meta_info=meta_info, return_feats=True)
        audio_feats, audio_preds = self.audio_branch.simple_test(
            audio=audio, meta_info=meta_info, return_feats=True)
        text_feats, text_preds = self.text_branch.simple_test(
            text=text, meta_info=meta_info, return_feats=True)
        feats_dict = dict(image=image_feats, audio=audio_feats, text=text_feats, video=video_feats)
        all_preds['fusion'] = self.fusion_head.simple_test(feats_dict, meta_info)
        all_preds.update(video_preds[0])
        all_preds.update(image_preds[0])
        all_preds.update(audio_preds[0])
        all_preds.update(text_preds[0])
        return [all_preds]