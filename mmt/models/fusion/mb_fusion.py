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
    def __init__(self, video_branch, image_branch, head, pretrained):
        super(MultiBranchesFusionModel, self).__init__()
        self.video_branch = build_video_branch(video_branch)
        self.image_branch = build_image_branch(image_branch)
        self.head = build_head(head)
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
        losses = self.head.forward_train(video_feats, gt_labels)
        self.image_branch(image)
        return losses

    def simple_test(self, video, image):
        video_feats = self.video_branch(video)
        return self.head.simple_test(video_feats)
