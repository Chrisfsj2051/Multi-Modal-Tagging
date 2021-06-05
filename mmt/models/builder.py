from mmcv.utils import Registry, build_from_cfg
from torch import nn

FUSION = Registry('fusion')
FRAME = Registry('frame')
IMAGE = Registry('image')
TEXT = Registry('text')
HEAD = Registry('head')
LOSS = Registry('loss')


def build(cfg, registry, default_args=None):
    """Build a module.

    Args:
        cfg (dict, list[dict]): The config of modules, is is either a dict
            or a list of configs.
        registry (:obj:`Registry`): A registry the module belongs to.
        default_args (dict, optional): Default arguments to build the module.
            Defaults to None.
    Returns:
        nn.Module: A built nn module.
    """
    if isinstance(cfg, list):
        modules = [
            build_from_cfg(cfg_, registry, default_args) for cfg_ in cfg
        ]
        return nn.Sequential(*modules)
    else:
        return build_from_cfg(cfg, registry, default_args)


def build_frame_branch(cfg):
    return build(cfg, FRAME)


def build_text_branch(cfg):
    return build(cfg, TEXT)


def build_head(cfg):
    return build(cfg, HEAD)


def build_loss(cfg):
    return build(cfg, LOSS)


def build_image_branch(cfg):
    return build(cfg, IMAGE)


def build_model(cfg):
    return build(cfg, FUSION)
