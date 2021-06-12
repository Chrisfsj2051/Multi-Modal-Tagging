from third_party.bert.modeling import BertModel, BertConfig
import torch.nn as nn

class Bert(nn.Module):
    def __init__(self):
        super(Bert, self).__init__()
        self.bert = BertModel.from


if __name__=='__main__':
    Bert()