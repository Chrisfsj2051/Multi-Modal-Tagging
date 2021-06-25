import random

import numpy as np
import torch

from mmt.models.builder import ARCH, build_head, build_model
from mmt.models.fusion import BaseFusionModel


@ARCH.register_module()
class VideoTextWithExtraTaskModel(BaseFusionModel):
    def __init__(self, branch_config, fusion_config, modal_match_config):
        super(VideoTextWithExtraTaskModel, self).__init__()
        self.video_branch = build_model(branch_config['video'])
        self.text_branch = build_model(branch_config['text'])
        self.fusion_head = build_head(fusion_config)
        self.match_head = build_head(modal_match_config)

    def forward_train(self, video, text, meta_info, gt_labels):
        video_feats, video_loss = self.video_branch.forward_train(
            video=video, meta_info=meta_info, gt_labels=gt_labels, return_feats=True)
        text_feats, text_loss = self.text_branch.forward_train(
            text=text, meta_info=meta_info, gt_labels=gt_labels, return_feats=True)
        feats_dict = dict(text=text_feats, video=video_feats)
        fusion_loss = self.fusion_head.forward_train(feats_dict, meta_info, gt_labels)
        bs = video.shape[0]
        assert bs >= 2
        match_gt_labels = gt_labels[0].new_tensor([1] * (bs // 2) + [0] * (bs // 2))
        text_inputs_id = []
        for i, gt in enumerate(match_gt_labels):
            if gt == 1:
                text_inputs_id.append(i)
            else:
                j = random.randint(0, bs - 1)
                while j == i:
                    j = random.randint(0, bs - 1)
                text_inputs_id.append(j)
        text_inputs_id = gt_labels[0].new_tensor(text_inputs_id)
        text_inputs = text_feats[text_inputs_id]
        video_inputs = video_feats
        match_loss = self.match_head.forward_train((video_inputs, text_inputs), meta_info, match_gt_labels[..., None])

        losses = {}
        for name, los in zip(
                ['video', 'text', 'fusion', 'modal_match'],
                [video_loss, text_loss, fusion_loss, match_loss]):  # noqa
            for key in los.keys():
                losses[name + '_' + key] = los[key]
        return losses

    def simple_test(self, video, image, text, audio, meta_info):
        all_preds = {}
        video_feats, video_preds = self.video_branch.simple_test(
            video=video, meta_info=meta_info, return_feats=True)
        text_feats, text_preds = self.text_branch.simple_test(
            text=text, meta_info=meta_info, return_feats=True)
        feats_dict = dict(text=text_feats, video=video_feats)
        all_preds['fusion'] = self.fusion_head.simple_test(feats_dict, meta_info)
        all_preds.update(video_preds[0])
        all_preds.update(text_preds[0])
        return [all_preds]
