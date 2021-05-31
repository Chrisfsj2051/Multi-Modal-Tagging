import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, vocab_size, ebd_dim, channel, filter_size):
        super(Model, self).__init__()
        self.embedding = nn.Embedding(vocab_size, ebd_dim)
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, channel, (k, ebd_dim)) for k in filter_size])
        self.fc = nn.Linear(len(filter_size) * len(config.filter_sizes), config.num_classes)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        out = self.embedding(x[0])
        out = out.unsqueeze(1)
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out = self.dropout(out)
        out = self.fc(out)
        return out