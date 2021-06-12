from mmt.datasets.pipelines.transforms import BertTokenize
from mmt.utils.third_party.bert_pytorch.pytorch_pretrained import BertModel, BertConfig

import torch.nn as nn

class Bert(nn.Module):
    def __init__(self, ckpt_path='pretrained/bert/'):
        super(Bert, self).__init__()
        self.bert = BertModel.from_pretrained(ckpt_path)

    # def forward(self, ):


if __name__=='__main__':
    # Bert()
    # if __name__ == '__main__':
    pipeline = BertTokenize('pretrained/bert')
    print(pipeline('江南皮革厂倒闭了！hello'))