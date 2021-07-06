import argparse

import mmcv
import torch
from pytorch_pretrained_vit import ViT
from torch.utils.data import DataLoader
import os

from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
import numpy as np
from tqdm import tqdm
import mmcv


def option():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', type=str, required=True)
    parser.add_argument('--save_path', type=str, required=True)
    return parser.parse_args()


def main():
    args = option()

    transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5),
    ])
    dataset = ImageFolder(args.video_path, transform)
    data_loader = DataLoader(dataset, batch_size=4, num_workers=16, shuffle=False)
    model = ViT('L_16_imagenet1k', pretrained=True)
    # model = ViT('L_16_imagenet1k', pretrained=True)
    del model.fc
    model.eval()
    model.cuda()
    idx_to_class = {v:k for k, v in dataset.class_to_idx.items()}
    feats_list = {k:[] for k in dataset.class_to_idx.keys()}
    for data in tqdm(data_loader):
        img = data[0].cuda()
        with torch.no_grad():
            # no LayerNorm?
            feats = model(img)
            feats = model.norm(feats)[:, 0]


        filenames = [idx_to_class[x] for x in data[1].tolist()]
        for i in range(len(filenames)):
            feats_list[filenames[i]].append(feats[i].tolist())

    for key, val in feats_list.items():
        val = np.array(val)
        mmcv.mkdir_or_exist(args.save_path)
        np.save(os.path.join(args.save_path, key+'.npy'), val)

if __name__ == '__main__':
    # t  = np.load('dataset/tagging/tagging_dataset_train_5k/video_npy/Youtube8M/tagging/6f718a69d07253a1aa5f4ea6aa5934d1.npy')
    # print(t.shape)
    main()
