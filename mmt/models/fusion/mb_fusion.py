import random

import numpy as np
import torch
import torch.nn as nn
from mmcv.runner import _load_checkpoint, load_state_dict

from mmt.models.builder import (FUSION, build_frame_branch, build_head,
                                build_image_branch, build_text_branch)
from mmt.models.fusion import BaseFusionModel
from mmt.utils import get_root_logger


@FUSION.register_module()
class MultiBranchesFusionModel(BaseFusionModel):
    """
    Args:
        mode (int):
            mode1: Train branches;
            mode2: Train fusion head;
            mode3: Train All.
    """
    def __init__(self, mode, modal_used, branch_config, head_config,
                 use_batch_norm, pretrained, modal_dropout_p):
        super(MultiBranchesFusionModel, self).__init__()
        build_branch_method = {
            'video': build_frame_branch,
            'image': build_image_branch,
            'text': build_text_branch,
            'audio': build_frame_branch
        }
        self.mode = mode
        self.modal_list = modal_used
        self.modal_dropout_p = modal_dropout_p
        # self.add_module('attn', build_head(attn_config))
        for modal in self.modal_list:
            self.add_module(f'{modal}_branch',
                            build_branch_method[modal](branch_config[modal]))
            # self.add_module(f'{modal}_ebd', build_head(ebd_config[modal]))
            self.add_module(f'{modal}_head', build_head(head_config[modal]))
            if use_batch_norm:
                self.add_module(f'{modal}_bn',
                                nn.LayerNorm(head_config[modal]['in_dim']))
        assert 'fusion' in head_config.keys()
        self.add_module('fusion_head', build_head(head_config['fusion']))
        self.use_batch_norm = use_batch_norm
        if pretrained and 'video' in pretrained and 'video' in modal_used:
            self.load_pretrained(self.video_branch, pretrained['video'])
        if pretrained and 'text' in pretrained and 'text' in modal_used:
            self.load_pretrained(self.text_branch, pretrained['text'])
        if pretrained and 'audio' in pretrained and 'text' in modal_used:
            self.load_pretrained(self.audio_branch, pretrained['audio'])
        if pretrained and 'image' in pretrained and 'image' in modal_used:
            # trans_key = None
            # if 'ResNet' in branch_config['image']['type']:
            #     trans_key = resnet_trans_key
            self.load_pretrained(self.image_branch, pretrained['image'])

        if mode == 1:
            for param in self.fusion_head.parameters():
                param.requires_grad = False
            # for param in self.attn.parameters():
            #     param.requires_grad = False
        elif mode == 2:
            for modal in self.modal_list:
                for arch in ('branch', 'head'):
                    for param in self.__getattr__(
                            f'{modal}_{arch}').parameters():
                        param.requires_grad = False

    def load_pretrained(self, model, pretrained):
        logger = get_root_logger()
        checkpoint = _load_checkpoint(pretrained)
        # OrderedDict is a subclass of dict
        if not isinstance(checkpoint, dict):
            raise RuntimeError(
                f'No state_dict found in checkpoint file {pretrained}')
        # get state_dict from checkpoint
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        # strip prefix of state_dict
        if list(state_dict.keys())[0].startswith('module.'):
            state_dict = {k[7:]: v for k, v in state_dict.items()}
        # load state_dict
        load_state_dict(model, state_dict, strict=False, logger=logger)

    def forward_train(self, video, image, text, audio, meta_info, gt_labels):
        if self.mode == 2:
            for modal in self.modal_list:
                for arch in ('branch', 'head'):
                    self.__getattr__(f'{modal}_{arch}').eval()

        feats_list, losses = [], {}
        modal_inputs = {
            'video': video,
            'image': image,
            'text': text,
            'audio': audio
        }
        for modal in self.modal_list:
            inputs = modal_inputs[modal]
            if modal == 'text':
                feats = self.__getattr__(f'{modal}_branch')(inputs, meta_info)
            else:
                feats = self.__getattr__(f'{modal}_branch')(inputs)
            if self.use_batch_norm:
                feats = self.__getattr__(f'{modal}_bn')(feats)
            # ebd = self.__getattr__(f'{modal}_ebd')(feats)
            feats_list.append(feats)
            if self.mode != 2:
                modal_loss = self.__getattr__(f'{modal}_head').forward_train(
                    feats, gt_labels)
                for key, val in modal_loss.items():
                    losses[f'{modal}_{key}'] = val

        if self.mode == 1:
            return losses
        if self.modal_dropout_p is not None:
            feats_list = self.apply_modal_dropout(feats_list)
        feats = torch.cat(feats_list, 1)
        # attn = self.attn(ebd)
        fusion_loss = self.fusion_head.forward_train(feats, gt_labels)
        for key, val in fusion_loss.items():
            losses[f'fusion_{key}'] = val
        return losses

    def apply_modal_dropout(self, modal_inputs):
        dropout_p = [[
            1 - self.modal_dropout_p[x]
            for _ in range(modal_inputs[0].shape[0])
        ] for x in self.modal_list]
        mask = np.random.binomial(1, dropout_p)
        for i in range(mask.shape[1]):
            if sum(mask[:, i]) == 0:
                mask[random.randint(0, mask.shape[0] - 1), i] = 1
        mask = torch.from_numpy(mask).cuda()
        outs = [x * y[..., None] for (x, y) in zip(modal_inputs, mask)]
        return outs

    def simple_test(self, video, image, text, audio, meta_info):
        ebd_list = []
        modal_inputs = {
            'video': video,
            'image': image,
            'text': text,
            'audio': audio
        }
        test_results = [{}]
        for modal in self.modal_list:
            inputs = modal_inputs[modal]
            if modal == 'text':
                feats = self.__getattr__(f'{modal}_branch')(inputs, meta_info)
            else:
                feats = self.__getattr__(f'{modal}_branch')(inputs)
            if self.use_batch_norm:
                feats = self.__getattr__(f'{modal}_bn')(feats)
            # ebd = self.__getattr__(f'{modal}_ebd')(feats)
            ebd_list.append(feats)
            test_results[0][modal] = self.__getattr__(
                f'{modal}_head').simple_test(feats)
        if self.mode == 1:
            return test_results
        ebd = torch.cat(ebd_list, 1)
        # attn = self.attn(ebd)
        test_results[0]['fusion'] = self.fusion_head.simple_test(ebd)
        return test_results
