import torch.nn as nn

from src.utils.mmt import BertTokenize
from src.utils.mmt.models import BACKBONE
from src.utils.mmt.utils import BertModel


@BACKBONE.register_module()
class Bert(nn.Module):
    def __init__(self, ckpt_path='pretrained/bert/', fixed=False):
        super(Bert, self).__init__()
        self.bert = BertModel.from_pretrained(ckpt_path)
        if fixed:
            self.bert.eval()
            for param in self.bert.parameters():
                param.requires_grad = False
        self.fixed = fixed

    def train(self, mode=True):
        """Convert the model into training mode while keep normalization layer
        freezed."""
        super(Bert, self).train(mode)
        if self.fixed:
            self.bert.eval()

    def forward(self, x, meta_info):
        assert x.ndim == 2
        # x_mask = x.new_tensor([item['text_mask'] for item in meta_info])
        # _, feat = self.bert(x,
        #                     attention_mask=x_mask,
        #                     output_all_encoded_layers=False)
        # return feat
        ocr, asr = x.split(x.shape[1] // 2, dim=1)
        infos = [[] for _ in range(4)]
        keys = ['ocr_seq_len', 'asr_seq_len', 'ocr_mask', 'asr_mask']
        for item in meta_info:
            for i, key in enumerate(keys):
                infos[i].append(item[key])
        for i in range(4):
            infos[i] = ocr.new_tensor(infos[i])
        _, ocr_feat = self.bert(ocr,
                                attention_mask=infos[2],
                                output_all_encoded_layers=False)

        _, asr_feat = self.bert(asr,
                                attention_mask=infos[3],
                                output_all_encoded_layers=False)
        return (asr_feat + ocr_feat) / 2


if __name__ == '__main__':
    # Bert()
    # if __name__ == '__main__':
    pipeline = BertTokenize('pretrained/bert', max_length=30)
    print(pipeline('江南皮革厂倒闭了！hello'))
