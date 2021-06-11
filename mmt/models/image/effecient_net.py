import torch

from mmt.models.builder import IMAGE
from mmt.models.image.effecient_net_pytorch.efficientnet_pytorch import \
    EfficientNet
import torch.nn as nn

@IMAGE.register_module()
class EffecientNet(nn.Module):
    print('in')

# global_params
if __name__ == '__main__':
    print('in')

    model = EfficientNet.from_pretrained('efficientnet-b0',
                                         include_top=False,
                                         load_fc=False).cuda()
    input = torch.randn((2, 3, 224, 224)).cuda()
    model._global_params.include_top = False
