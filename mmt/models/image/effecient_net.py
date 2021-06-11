import torch

from mmt.models.image.effecient_net_pytorch.efficientnet_pytorch import \
    EfficientNet

# global_params
if __name__ == '__main__':
    model = EfficientNet.from_pretrained('efficientnet-b0',
                                         include_top=False).cuda()
    input = torch.randn((2, 3, 224, 224)).cuda()
    model._global_params.include_top = False
    print('in')
