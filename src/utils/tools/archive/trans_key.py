import argparse

import torch

ckpt_path = 'pretrained/resnet50_batch256_imagenet_20200708-cfb998bf.pth'
ckpt = torch.load(ckpt_path)
ret = {}
for key, val in ckpt['state_dict'].items():
    ret[key.replace('backbone.', '')] = val

torch.save(ret, ckpt_path)
