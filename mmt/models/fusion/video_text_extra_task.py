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

    def apply_modal_dropout(self, feats_dict):
        key_list, item_list = [], []
        for key, val in feats_dict.items():
            key_list.append(key)
            item_list.append(val)

        bs = item_list[0].shape[0]
        dropout_p = [[1 - self.modal_dropout_p[x] for _ in range(bs)]
                     for x in self.modal_list]
        mask = np.random.binomial(1, dropout_p)
        for i in range(mask.shape[1]):
            if sum(mask[:, i]) == 0:
                mask[random.randint(0, mask.shape[0] - 1), i] = 1
        mask = torch.from_numpy(mask).cuda()
        item_list = [x * y[..., None] for (x, y) in zip(item_list, mask)]
        for i, k in enumerate(key_list):
            feats_dict[k] = item_list[i]
        return feats_dict

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
