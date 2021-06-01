import torch
import torch.nn as nn
import torch.nn.functional as F

from mmt.models.builder import TEXT
from mmt.utils.tokenization import FullTokenizer

@TEXT.register_module()
class TextCNN(nn.Module):
    def __init__(self, vocab_size, ebd_dim, channel_in, channel_out, filter_size):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, ebd_dim)
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, channel_in, (k, ebd_dim)) for k in filter_size])
        self.fc = nn.Linear(channel_in * len(filter_size), channel_out)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        assert x.max().item() < self.embedding.shape[0]
        out = self.embedding(x)
        out = out.unsqueeze(1)
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out = self.fc(out)
        return out

@TEXT.register_module()
class TwoStreamTextCNN(nn.Module):
    def forward(self, x):
        print('inininini')
        ocr, asr = x.split()
        out = self.embedding(x)
        out = out.unsqueeze(1)
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out = self.fc(out)
        return out


if __name__ == '__main__':
    model = TextCNN(9905, 300, 256, 1024, (2, 3, 4))
    tok = FullTokenizer('dataset/vocab_small.txt')
    text = 'hello world 我来了|哈哈哈'
    token = tok.tokenize(text)
    inputs = tok.convert_tokens_to_ids(token)
    pad_token = tok.convert_tokens_to_ids(['[PAD]'])
    inputs = inputs + pad_token * (128 - len(inputs))
    inputs = torch.LongTensor(inputs)[None]
    model(inputs)

