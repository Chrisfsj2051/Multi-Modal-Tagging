import random

import numpy as np
import torch

from mmt.models.builder import ARCH, build_head, build_model
from mmt.models.fusion import BaseFusionModel


@ARCH.register_module()
class MultiBranchFusionModel(BaseFusionModel):
    def __init__(self, branch_config, fusion_config, modal_dropout_p):
        super(MultiBranchFusionModel, self).__init__()
        self.video_branch = build_model(branch_config['video'])
        self.audio_branch = build_model(branch_config['audio'])
        self.image_branch = build_model(branch_config['image'])
        self.text_branch = build_model(branch_config['text'])
        self.fusion_head = build_head(fusion_config)
        self.modal_dropout_p = modal_dropout_p

    def forward_train(self, video, image, text, audio, meta_info, gt_labels):
        video_feats, video_loss = self.video_branch.forward_train(
            video, meta_info, gt_labels)
        image_feats, image_loss = self.image_branch.forward_train(
            image, meta_info, gt_labels)
        audio_feats, audio_loss = self.audio_branch.forward_train(
            audio, meta_info, gt_labels)
        text_feats, text_loss = self.text_branch.forward_train(
            text, meta_info, gt_labels)
        feats_dict = dict(image=image_feats,
                          audio=audio_feats,
                          text=text_feats,
                          video=video_feats)
        fusion_loss = self.fusion_head.forward_train(feats_dict, meta_info,
                                                     gt_labels)
        losses = {}
        for name, los in zip(
            ['video', 'image', 'audio', 'text', 'fusion'],
            [video_loss, image_loss, audio_loss, text_loss, fusion_loss
             ]):  # noqa
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
        pass
