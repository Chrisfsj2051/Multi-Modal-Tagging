import torch
import torch.nn as nn
import torch.nn.functional as F

from mmt.models.builder import TEXT
from mmt.utils.tokenization import FullTokenizer


@TEXT.register_module()
class TextCNN(nn.Module):
    def __init__(self,
                 vocab_size,
                 ebd_dim,
                 channel_in,
                 channel_out,
                 filter_size,
                 dropout_p=None):
        super(TextCNN, self).__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, ebd_dim)
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, channel_in, (k, ebd_dim)) for k in filter_size])
        self.fc = nn.Linear(channel_in * len(filter_size), channel_out)
        self.use_dropout = dropout_p is not None
        if self.use_dropout:
            self.dropout = nn.Dropout(p=dropout_p)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x, meta_info):
        assert x.max().item() < self.vocab_size
        out = self.embedding(x)
        out = out.unsqueeze(1)
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs],
                        1)
        out = self.fc(out)
        if self.use_dropout and self.training:
            out = self.dropout(out)
        return out


@TEXT.register_module()
class TwoStreamTextCNN(TextCNN):
    def forward(self, x, meta_info):
        assert x.ndim == 2
        ocr, asr = x.split(x.shape[1] // 2, dim=1)
        ocr_feat = super(TwoStreamTextCNN, self).forward(ocr, meta_info)
        asr_feat = super(TwoStreamTextCNN, self).forward(asr, meta_info)
        out = (ocr_feat + asr_feat) / 2
        return out
        # return super(TwoStreamTextCNN, self).forward(x)


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
