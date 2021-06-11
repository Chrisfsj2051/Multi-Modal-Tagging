import torch
import torch.nn as nn

from mmt.models.builder import IMAGE
from mmt.models.image.effecient_net_pytorch.efficientnet_pytorch import \
    EfficientNet as ThirdPartyEfficientNet


@IMAGE.register_module()
class EffecientNet(nn.Module):
    def __init__(self, arch):
        super(EffecientNet, self).__init__()
        self.net = ThirdPartyEfficientNet.from_pretrained(arch,
                                                          include_top=False,
                                                          load_fc=False)

    def forward(self, x):
        assert ((x.shape[2] == self.net._global_params.image_size) and
                (x.shape[3] == self.net._global_params.image_size)), \
            f'model image size={self.net._global_params.image_size}'

        outs = self.net(x)
        assert outs.shape[-1] == outs.shape[-2] == 1
        outs = outs[:, :, 0, 0]
        return outs


# global_params
if __name__ == '__main__':
    print('in')

    model = ThirdPartyEfficientNet.from_pretrained('efficientnet-b0',
                                                   include_top=False,
                                                   load_fc=False).cuda()

    input = torch.randn((2, 3, 224, 224)).cuda()
    model._global_params.include_top = False
