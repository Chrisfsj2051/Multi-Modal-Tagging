import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_norm_layer
from torch.nn import init

from mmt.datasets.pipelines.transforms import VideoResamplePad
from mmt.models.builder import BACKBONE


# based on https://github.com/lyakaap/NetVLAD-pytorch/blob/master/netvlad.py
@BACKBONE.register_module()
class NeXtVLAD(nn.Module):
    """NetVLAD layer implementation."""
    def __init__(self,
                 feature_size,
                 max_frames,
                 cluster_size,
                 norm_cfg=dict(type='BN1d'),
                 expansion=2,
                 groups=16):
        super(NeXtVLAD, self).__init__()
        self.cluster_size = cluster_size
        self.feature_size = feature_size
        self.max_frames = max_frames
        self.expansion = expansion
        self.groups = groups
        self.w1 = nn.Parameter(
            torch.randn((feature_size, expansion * feature_size)))
        self.w2 = nn.Parameter(torch.randn((feature_size * expansion, groups)))
        self.w3 = nn.Parameter(
            torch.randn((expansion * feature_size, groups * cluster_size)))
        self.w4 = nn.Parameter(
            torch.randn(1, self.expansion * self.feature_size // self.groups,
                        cluster_size))

        self.bn1 = build_norm_layer(norm_cfg, groups * cluster_size)[1]
        self.bn2 = build_norm_layer(
            norm_cfg, self.cluster_size * self.expansion * self.feature_size //
            self.groups)[1]
        self.init_parameters()

    def init_parameters(self):
        for layer in (self.w1, self.w2, self.w3, self.w4):
            init.kaiming_uniform_(layer, mode='fan_in')

    def forward(self, input, meta_info):
        input = torch.matmul(input, self.w1)
        attention = torch.matmul(input, self.w2).sigmoid()
        attention = attention.reshape([-1, self.max_frames * self.groups, 1])
        reshaped_input = input.reshape(
            [-1, self.expansion * self.feature_size])
        activation = torch.matmul(reshaped_input, self.w3)
        activation = self.bn1(activation)
        activation = activation.reshape(
            [-1, self.max_frames * self.groups, self.cluster_size])
        activation = activation.softmax(axis=-1)
        activation = activation * attention
        a_sum = torch.mean(activation, -2, keepdim=True)
        a = a_sum * self.w4
        activation = activation.permute((0, 2, 1))
        feature_size = self.expansion * self.feature_size // self.groups
        reshaped_input = input.reshape(
            [-1, self.max_frames * self.groups, feature_size])
        vlad = torch.matmul(activation, reshaped_input)
        vlad = vlad.permute((0, 2, 1))
        vlad = vlad - a
        vlad = F.normalize(vlad, p=2, dim=1)
        vlad = vlad.reshape([-1, self.cluster_size * feature_size])
        vlad = self.bn2(vlad)
        return vlad

@BACKBONE.register_module()
class MultiScaleNeXtVLAD(nn.Module):
    def __init__(self,
                 feature_size,
                 max_frames_list,
                 cluster_size,
                 norm_cfg=dict(type='BN1d'),
                 expansion=2,
                 groups=16):
        super(MultiScaleNeXtVLAD, self).__init__()
        self.next_vlads = nn.ModuleList()
        for max_frames in max_frames_list:
            self.next_vlads.append(
                NeXtVLAD(
                    feature_size,
                    max_frames,
                    cluster_size,
                    norm_cfg=norm_cfg,
                    expansion=expansion,
                    groups=groups
                )
            )
        self.max_frames_list = max_frames_list

    def forward(self, input, meta_info):
        results = []
        for i, max_frames in enumerate(self.max_frames_list):
            video_list = []
            for video, meta in zip(input, meta_info):
                video = video.cpu().numpy()
                video = VideoResamplePad.resample(
                    video, meta_info[0]['video_len'],  max_frames)
                video_list.append(input.new_tensor(video)[None])

            video = torch.cat(video_list, 0)
            results.append(self.next_vlads[i](video, meta_info))

        return torch.cat(results, 1)