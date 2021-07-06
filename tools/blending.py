import argparse
import os
import warnings

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)

from mmt import build_dataloader, build_dataset
from mmt.apis import multi_gpu_test, single_gpu_test
from mmt.datasets.utils import replace_ImageToTensor
from mmt.models.builder import build_model


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config_1', help='test config file path')
    parser.add_argument('config_2', help='test config file path')
    parser.add_argument('checkpoint_1', help='checkpoint file')
    parser.add_argument('checkpoint_2', help='checkpoint file')
    parser.add_argument('--launcher',
                        choices=['none', 'pytorch', 'slurm', 'mpi'],
                        default='none',
                        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.eval_options:
        raise ValueError(
            '--options and --eval-options cannot be both '
            'specified, --options is deprecated in favor of --eval-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --eval-options')
        args.eval_options = args.options
    return args


def build_from_config(config, ckpt, distributed):
    cfg = Config.fromfile(config)
    dataset = build_dataset(cfg.data.train)
    model = build_model(cfg.model)
    load_checkpoint(model, ckpt, map_location='cpu')
    return model, dataset


def main():
    args = parse_args()
    # init distributed env first, since logger depends on the dist info.
    distributed = False
    model_1, dataset_1 = build_from_config(args.config_1, args.checkpoint_1, distributed)
    model_2, dataset_2 = build_from_config(args.config_1, args.checkpoint_1, distributed)



if __name__ == '__main__':
    main()
