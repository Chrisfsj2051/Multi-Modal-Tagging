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

    def forward_train(self, video, image, text, audio, meta_info, gt_labels, gt_labels_ignore=None):
        video_feats, video_loss = self.video_branch.forward_train(
            video=video, meta_info=meta_info, gt_labels=gt_labels,
            gt_labels_ignore=gt_labels_ignore, return_feats=True)
        image_feats, image_loss = self.image_branch.forward_train(
            image=image, meta_info=meta_info, gt_labels=gt_labels,
            gt_labels_ignore=gt_labels_ignore, return_feats=True)
        audio_feats, audio_loss = self.audio_branch.forward_train(
            audio=audio, meta_info=meta_info, gt_labels=gt_labels,
            gt_labels_ignore=gt_labels_ignore, return_feats=True)
        text_feats, text_loss = self.text_branch.forward_train(
            text=text, meta_info=meta_info, gt_labels=gt_labels,
            gt_labels_ignore=gt_labels_ignore, return_feats=True)
        feats_dict = dict(image=image_feats, audio=audio_feats, text=text_feats, video=video_feats)
        feats_dict = self.apply_modal_dropout(feats_dict)
        fusion_loss = self.fusion_head.forward_train(feats_dict, meta_info, gt_labels, gt_labels_ignore)
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
                     for x in key_list]
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


@ARCH.register_module()
class MultiBranchFusionModelWithModalMatch(MultiBranchFusionModel):

    def __init__(self, *args, modal_match_config, **kwargs):
        super(MultiBranchFusionModelWithModalMatch, self).__init__(*args, **kwargs)
        self.match_head = build_head(modal_match_config)

    def modal_match_forward_train(self, video, text, meta_info, gt_labels):
        video_feats = self.video_branch.backbone(video, meta_info=meta_info)
        text_feats = self.text_branch.backbone(text, meta_info=meta_info)
        feats_list = [video_feats, text_feats]
        gt_labels = torch.cat([x[None] for x in gt_labels], 0)
        modal_match_loss = self.match_head.forward_train(feats_list, meta_info, gt_labels)
        return modal_match_loss

    def modal_match_simple_test(self, video, text, meta_info):
        video_feats = self.video_branch.backbone(video, meta_info=meta_info)
        text_feats = self.text_branch.backbone(text, meta_info=meta_info)
        feats_list = [video_feats, text_feats]
        return self.match_head.simple_test(feats_list, meta_info)

    def forward_train(self, video, image, text, audio, extra, meta_info, gt_labels):
        labeled_loss = super(MultiBranchFusionModelWithModalMatch, self).forward_train(
            video, image, text, audio, meta_info, gt_labels
        )
        modal_match_loss = self.modal_match_forward_train(
            extra['video'], extra['text'], extra['meta_info'], extra['gt_labels']
        )
        for key, val in modal_match_loss.items():
            labeled_loss[f'modal_match_{key}'] = val
        return labeled_loss

    def simple_test(self, video, image, text, audio, meta_info):
        assert len(meta_info) == 1
        if meta_info[0].pop('modal_match', False):
            return self.modal_match_simple_test(video, text, meta_info)
        else:
            return super(MultiBranchFusionModelWithModalMatch, self).simple_test(
                video, image, text, audio, meta_info)
