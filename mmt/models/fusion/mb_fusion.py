import torch
from mmcv.runner import _load_checkpoint, load_checkpoint, load_state_dict

from mmt.models.builder import (FUSION, build_head, build_image_branch,
                                build_video_branch)
from mmt.models.fusion import BaseFusionModel
from mmt.utils import get_root_logger


def resnet_trans_key(ckpt):
    ret = dict()
    for key, val in ckpt.items():
        ret[key.replace('backbone.', '')] = val
    return ret


@FUSION.register_module()
class MultiBranchesFusionModel(BaseFusionModel):
    """Base class for detectors."""
    def __init__(self, video_branch, image_branch, video_edb, image_edb, video_head, image_head, fusion_head, pretrained):
        super(MultiBranchesFusionModel, self).__init__()
        self.video_branch = build_video_branch(video_branch)
        self.image_branch = build_image_branch(image_branch)
        self.image_edb = build_head(image_edb)
        self.video_edb = build_head(video_edb)
        self.video_head = build_head(video_head)
        self.image_head = build_head(image_head)
        self.fusion_head = build_head(fusion_head)
        if pretrained and 'video' in pretrained:
            self.load_pretrained(self.video_branch, pretrained['viedo'])
        if pretrained and 'image' in pretrained:
            trans_key = None
            if 'ResNet' in image_branch['type']:
                trans_key = resnet_trans_key
            self.load_pretrained(self.image_branch, pretrained['image'],
                                 trans_key)

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

    def forward_train(self, video, image, gt_labels):
        video_feats = self.video_branch(video)
        image_feats = self.image_branch(image)

        video_ebd = self.video_edb(video_feats)
        image_ebd = self.image_edb(image_feats)

        video_loss = self.video_head.forward_train(video_ebd, gt_labels)
        image_loss = self.image_head.forward_train(image_ebd, gt_labels)

        ebd = torch.cat([video_ebd, image_ebd], 1)
        fusion_loss = self.fusion_head.forward_train(ebd, gt_labels)

        loss_dict = {'video_loss': video_loss,
                     'image_loss': image_loss,
                     'fusion_loss': fusion_loss}

        return loss_dict

    def simple_test(self, video, image):
        video_feats = self.video_branch(video)
        image_feats = self.image_branch(image)

        video_ebd = self.video_edb(video_feats)
        image_ebd = self.image_edb(image_feats)
        ebd = torch.cat([video_ebd, image_ebd], 1)

        return self.fusion_head.simple_test(ebd)
