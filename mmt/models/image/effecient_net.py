import mmcv
import torch.nn as nn

from mmt.datasets.pipelines.formating import to_tensor
from mmt.models.builder import BACKBONE
from mmt.models.image.effecient_net_pytorch.efficientnet_pytorch import \
    EfficientNet as ThirdPartyEfficientNet


@BACKBONE.register_module()
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
    import numpy as np
    model = ThirdPartyEfficientNet.from_pretrained('efficientnet-b0',
                                                   include_top=True,
                                                   load_fc=True).cuda()

    inputs = mmcv.imread('aux_files/img.png')
    mean = [123.675, 116.28, 103.53]
    std = [58.395, 57.12, 57.375]
    # std=[0.229*255, 0.224*255, 0.225*255]
    # mean=[103.53, 116.28, 123.675]
    # std=[57.375, 57.12, 58.395]
    # mean.reverse()
    # std.reverse()
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    print(mean, std)
    inputs = mmcv.imresize(inputs, (224, 224))
    inputs = mmcv.imnormalize(inputs, mean, std).transpose(2, 0, 1)
    inputs = to_tensor(inputs).cuda()[None]
    model.eval()

    # from PIL import Image
    # import torch
    # from torchvision import transforms
    # tfms = transforms.Compose([transforms.Resize((224, 224)),
    # transforms.ToTensor(),
    #                            transforms.Normalize([0.485, 0.456, 0.406],
    #                                                 [0.229, 0.224, 0.225])
    #                            ]
    #                           )
    # inputs = tfms(Image.open('aux_files/img.png'
    # ).convert('RGB')).unsqueeze(0)
    # 0.6372

    preds = model(inputs.cuda()).softmax(1).max()
    print(preds)
    print('Done')
