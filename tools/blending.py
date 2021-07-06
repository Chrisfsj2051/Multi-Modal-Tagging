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
    parser.add_argument('config_blending', help='test config file path')
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
    return args


def build_from_config(config, ckpt, distributed):
    cfg = Config.fromfile(config)
    model = build_model(cfg.model)
    load_checkpoint(model, ckpt, map_location='cpu')
    model = MMDataParallel(model, device_ids=[0])
    model.eval()
    return model


def main():
    args = parse_args()
    # init distributed env first, since logger depends on the dist info.
    distributed = False
    model_1 = build_from_config(args.config_1, args.checkpoint_1, distributed)
    model_2 = build_from_config(args.config_1, args.checkpoint_1, distributed)
    blending_cfg = Config.fromfile(args.config_blending)
    train_ds = build_dataset(blending_cfg.data.train)
    val_ds = build_dataset(blending_cfg.data.val)
    model =
    train_loader = build_dataloader(train_ds, samples_per_gpu=1, workers_per_gpu=1, dist=False, shuffle=False)
    val_loader = build_dataloader(val_ds, samples_per_gpu=1, workers_per_gpu=1, dist=False, shuffle=False)
    for data in train_loader:
        print('in')
    feat_1, feat_2 = [], []
    # for data in train_ds:
    #     print('')



if __name__ == '__main__':
    main()

