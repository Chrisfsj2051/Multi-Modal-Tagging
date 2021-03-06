import argparse
import os
from multiprocessing import Pool

import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import (load_checkpoint)
from sklearn.ensemble import GradientBoostingClassifier
from tqdm import tqdm

from src.utils.mmt import build_dataloader, build_dataset
from src.utils.mmt.models import build_model


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


def get_feat(feat, idx):
    # pos_idx = np.array([idx, 82+idx]).astype(np.int32)
    # return feat[:, idx]
    return feat


def fit_model(args):
    cls, feat, binary_labels_list = args
    binary_labels = binary_labels_list[cls]
    if all(binary_labels == 0) or all(binary_labels == 1):
        return None
    else:
        gbdt = GradientBoostingClassifier(learning_rate=0.02, subsample=0.5, max_depth=6, n_estimators=30)
        feat = get_feat(feat, cls)
        gbdt.fit(feat, binary_labels)
        return gbdt


def main():
    args = parse_args()
    # init distributed env first, since logger depends on the dist info.
    distributed = False
    model_1 = build_from_config(args.config_1, args.checkpoint_1, distributed)
    model_2 = build_from_config(args.config_1, args.checkpoint_1, distributed)
    blending_cfg = Config.fromfile(args.config_blending)
    train_ds = build_dataset(blending_cfg.data.train)
    val_ds = build_dataset(blending_cfg.data.val)
    train_ds.with_sigmoid = False
    val_ds.with_sigmoid = False
    train_loader = build_dataloader(train_ds, samples_per_gpu=1, workers_per_gpu=1, dist=False, shuffle=False)
    val_loader = build_dataloader(val_ds, samples_per_gpu=1, workers_per_gpu=1, dist=False, shuffle=False)
    feat_list_1, feat_list_2 = [], []
    for data in tqdm(train_loader):
        with torch.no_grad():
            pred_1 = model_1.forward(return_loss=False, **data)[0]['fusion'].sigmoid()
            pred_2 = model_2.forward(return_loss=False, **data)[0]['fusion'].sigmoid()
        feat_list_1.append(pred_1.cpu().numpy())
        feat_list_2.append(pred_2.cpu().numpy())

    feat_1 = np.concatenate(feat_list_1, 0)
    feat_2 = np.concatenate(feat_list_2, 0)
    feat = np.concatenate([feat_1, feat_2], 1)
    gt_labels = train_ds.gt_label

    # debug !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # debug !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # debug !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # debug !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # debug !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # gt_labels[0] = list(range(82))
    # gt_labels[1] = [81]
    # debug !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # debug !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # debug !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # debug !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # debug !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    pool = Pool(12)
    binary_labels_list = []

    for cls in range(82):
        binary_labels = [cls in lab for lab in gt_labels]
        binary_labels = np.array(binary_labels).astype(np.int32)
        binary_labels_list.append(binary_labels)

    gbdt = list(
        tqdm(
            pool.imap(fit_model, [[i, feat, binary_labels_list] for i in range(82)]),
            total=82
        ))
    pool.close()
    pool.join()
    preds = []
    for data in tqdm(val_loader):
        with torch.no_grad():
            pred_1 = model_1.forward(return_loss=False, **data)[0]['fusion'].sigmoid()
            pred_2 = model_2.forward(return_loss=False, **data)[0]['video'].sigmoid()
        feat_1 = pred_1.cpu().numpy()
        feat_2 = pred_2.cpu().numpy()
        feat = np.concatenate([feat_1, feat_2], 1)
        # prob = [gbdt[i].predict_proba(get_feat(feat, i))[0][1] if gbdt[i] else feat_1[0][i] for i in range(len(gbdt))]
        video_better = [1, 2, 3, 4, 5, 7, 9, 10, 11, 12, 14, 18, 20, 21, 22, 23, 29, 31, 34, 35, 36, 38, 41, 43, 44, 46, 47, 49, 50,
         52, 54, 55, 56, 57, 64, 66, 69, 71, 75]
        prob = [feat_1[0][i] if i not in video_better else feat_2[0][i] for i in range(82)]
        preds.append(prob)
    preds_fmt = [{'blending': [torch.FloatTensor(x)]} for x in preds]
    eval_res = val_ds.evaluate(preds_fmt)
    print(eval_res)


if __name__ == '__main__':
    main()
