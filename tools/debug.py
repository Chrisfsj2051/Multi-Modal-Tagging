# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F


'''Convolutional Neural Networks for Sentence Classification'''


class TextCNN(nn.Module):
    def __init__(self):
        super(TextCNN, self).__init__()
        # filter_sizes = list(map(int,config.net.textmodel.textcnn.filter_sizess.split(',')))
        filter_sizes = [1, 2, 3, 4]
        self.convs = nn.ModuleList(
            [nn.Conv2d(1,150, (k, 200)) for k in filter_sizes])
        # self.dropout = nn.Dropout(config.net.dropout)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        # out = self.embedding(x) #self.embedding(x[0])
        out = x.unsqueeze(1)
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        # out = self.dropout(out)
        #out = self.fc(out)
        return out




if __name__ == '__main__':
    import numpy as np
    import pdb
    # pdb.set_trace()

    # cfg = Config(dataset='tiku', embedding='')
    # cfg.n_vocab = 500001
    # cfg.embed = 200
    x = torch.randint(0, 500000, (8,25))
    model = TextCNN()
    model(x)
