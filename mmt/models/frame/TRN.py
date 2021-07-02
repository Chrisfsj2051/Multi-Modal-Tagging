import torch.nn as nn
import torch

from mmt.models.builder import BACKBONE


@BACKBONE.register_module()
class TRN(nn.Module):
    def __init__(self, num_segment, input_dim, output_dim):
        super(TRN, self).__init__()
        self.num_segment = num_segment
        self.layer1 = nn.Linear(input_dim, 768)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(num_segment * 768, 768)
        self.layer3 = nn.Linear(768, output_dim)
        self.layer4 = nn.Linear(input_dim, output_dim)

    def forward(self, input, meta_info):  # input : [batch,segment,RGB_FEATURE_SIZE or 128]
        aver_mean = torch.mean(input, 1)
        out1 = self.relu(self.layer1(input))
        out1 = self.relu(self.layer2(out1.view(-1, self.num_segment * 768)))
        out1 = self.layer3(out1)
        out2 = self.layer4(aver_mean)
        out = self.relu(out1 + out2)
        return out
