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
class SingleBranchesFusionModel(BaseFusionModel):
    """
    Args:
        mode (int):
            mode1: Train branches;
            mode2: Train fusion head;
            mode3: Train All.
    """
    def __init__(self, modal_used, branch_config,
                 head_config, use_batch_norm, pretrained):
        super(SingleBranchesFusionModel, self).__init__()
        assert len(modal_used) == 1
        build_branch_method = {
            'video': build_frame_branch,
            'image': build_image_branch,
            'text': build_text_branch,
            'audio': build_frame_branch
        }
        self.modal_list = modal_used
        for modal in self.modal_list:
            self.add_module(f'{modal}_branch',
                            build_branch_method[modal](branch_config[modal]))
            # self.add_module(f'{modal}_ebd', build_head(ebd_config[modal]))
            self.add_module(f'{modal}_head', build_head(head_config[modal]))
            if use_batch_norm:
                self.add_module(f'{modal}_bn',
                                nn.LayerNorm(head_config[modal]['in_dim']))
        self.use_batch_norm = use_batch_norm
        if pretrained and 'video' in pretrained and 'video' in modal_used:
            self.load_pretrained(self.video_branch, pretrained['video'])
        if pretrained and 'text' in pretrained and 'text' in modal_used:
            self.load_pretrained(self.text_branch, pretrained['text'])
        if pretrained and 'audio' in pretrained and 'text' in modal_used:
            self.load_pretrained(self.audio_branch, pretrained['audio'])
        if pretrained and 'image' in pretrained and 'image' in modal_used:
            self.load_pretrained(self.image_branch, pretrained['image'])


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

    def forward_train(self, video, image, text, audio, gt_labels):

        feats_list, losses = [], {}
        modal_inputs = {
            'video': video,
            'image': image,
            'text': text,
            'audio': audio
        }
        for modal in self.modal_list:
            inputs = modal_inputs[modal]
            feats = self.__getattr__(f'{modal}_branch')(inputs)
            if self.use_batch_norm:
                feats = self.__getattr__(f'{modal}_bn')(feats)
            feats_list.append(feats)
            modal_loss = self.__getattr__(f'{modal}_head').forward_train(
                feats, gt_labels)
            for key, val in modal_loss.items():
                losses[f'{modal}_{key}'] = val
        return losses


    def simple_test(self, video, image, text, audio):
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
            feats = self.__getattr__(f'{modal}_branch')(inputs)
            if self.use_batch_norm:
                feats = self.__getattr__(f'{modal}_bn')(feats)
            ebd_list.append(feats)
            test_results[0][modal] = self.__getattr__(f'{modal}_head').simple_test(feats)
        return test_results
