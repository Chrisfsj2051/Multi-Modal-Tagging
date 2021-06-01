import torch
from mmcv.runner import _load_checkpoint, load_checkpoint, load_state_dict

from mmt.models.builder import (FUSION, build_head, build_image_branch,
                                build_frame_branch, build_text_branch)
from mmt.models.fusion import BaseFusionModel
from mmt.utils import get_root_logger
import torch.nn as nn

def resnet_trans_key(ckpt):
    ret = dict()
    for key, val in ckpt.items():
        ret[key.replace('backbone.', '')] = val
    return ret


@FUSION.register_module()
class MultiBranchesFusionModel(BaseFusionModel):
    """
    Args:
        mode (int): mode1: Train branches; mode2: Train fusion head; mode3: Train All.
    """
    def __init__(self,
                 mode,
                 modal_used,
                 branch_config,
                 ebd_config,
                 head_config,
                 use_layer_norm,
                 pretrained):
        super(MultiBranchesFusionModel, self).__init__()
        build_branch_method = {'video': build_frame_branch,
                               'image': build_image_branch,
                               'text': build_text_branch,
                               'audio': build_frame_branch}
        self.mode = mode
        self.modal_list = modal_used
        for modal in self.modal_list:
            self.add_module(f'{modal}_branch',
                            build_branch_method[modal](branch_config[modal]))
            self.add_module(f'{modal}_ebd',
                            build_head(ebd_config[modal]))
            self.add_module(f'{modal}_head',
                            build_head(head_config[modal]))
        assert 'fusion' in head_config.keys()
        self.add_module('fusion_head', build_head(head_config['fusion']))
        self.use_layer_norm = use_layer_norm
        if self.use_layer_norm:
            self.fusion_layer_norm = nn.LayerNorm(head_config['fusion']['in_dim'])
        if pretrained and 'video' in pretrained:
            self.load_pretrained(self.video_branch, pretrained['viedo'])
        if pretrained and 'image' in pretrained:
            trans_key = None
            if 'ResNet' in branch_config['image']['type']:
                trans_key = resnet_trans_key
            self.load_pretrained(self.image_branch, pretrained['image'],
                                 trans_key)

        if mode == 1:
            for param in self.fusion_head.parameters():
                param.requires_grad = False
        elif mode == 2:
            for modal in self.modal_list:
                for arch in ('branch', 'ebd', 'head'):
                    for param in self.__getattr__(f'{modal}_{arch}').parameters():
                        param.requires_grad=False

    def load_pretrained(self, model, pretrained, trans_key=None):
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
        if trans_key:
            state_dict = trans_key(state_dict)
        # load state_dict
        load_state_dict(model, state_dict, strict=False, logger=logger)

    def forward_train(self, video, image, text, audio, gt_labels):
        if self.mode == 2:
            for modal in self.modal_list:
                for arch in ('branch', 'ebd', 'head'):
                    self.__getattr__(f'{modal}_{arch}').eval()

        ebd_list, losses = [], {}
        modal_inputs = {'video': video,
                        'image': image,
                        'text': text,
                        'audio': audio}
        for modal in self.modal_list:
            inputs = modal_inputs[modal]
            feats = self.__getattr__(f'{modal}_branch')(inputs)
            ebd_list.append(feats)
            if self.mode != 2:
                ebd = self.__getattr__(f'{modal}_ebd')(feats)
                losses[f'{modal}_loss'] = self.__getattr__(
                    f'{modal}_head').forward_train(ebd, gt_labels)
        if self.mode == 1:
            return losses
        ebd = torch.cat(ebd_list, 1)
        if self.use_layer_norm:
            ebd = self.fusion_layer_norm(ebd)
        losses['fusion_loss'] = self.fusion_head.forward_train(ebd, gt_labels)
        return losses

    def simple_test(self, video, image, text, audio):
        ebd_list, losses = [], {}
        modal_inputs = {'video': video,
                        'image': image,
                        'text': text,
                        'audio': audio}
        test_results = [{}]
        for modal in self.modal_list:
            inputs = modal_inputs[modal]
            feats = self.__getattr__(f'{modal}_branch')(inputs)
            ebd = self.__getattr__(f'{modal}_ebd')(feats)
            ebd_list.append(feats)
            test_results[0][modal] = self.__getattr__(f'{modal}_head')(ebd)
        ebd = torch.cat(ebd_list, 1)
        if self.use_layer_norm:
            ebd = self.fusion_layer_norm(ebd)
        test_results[0]['fusion'] = self.fusion_head.simple_test(ebd)
        return test_results
