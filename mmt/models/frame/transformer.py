import torch
import torch.nn as nn
import torch.nn.functional as F

from mmt.models.builder import BACKBONE, build_head


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, scale=None):
        '''
        Args:
            Q: [batch_size, len_Q, dim_Q]
            K: [batch_size, len_K, dim_K]
            V: [batch_size, len_V, dim_V]
            scale: 缩放因子 论文为根号dim_K
        Return:
            self-attention后的张量，以及attention张量
        '''
        attention = torch.matmul(Q, K.permute(0, 2, 1))
        if scale:
            attention = attention * scale
        # if mask:  # TODO change this
        #     attention = attention.masked_fill_(mask == 0, -1e9)
        attention = F.softmax(attention, dim=-1)
        context = torch.matmul(attention, V)
        return context


class MultiHeadAttention(nn.Module):
    def __init__(self, dim_in, num_head):
        super(MultiHeadAttention, self).__init__()
        self.num_head = num_head
        assert dim_in % num_head == 0
        dim_head = dim_in // num_head
        self.dim_head = dim_head
        self.W_Q = nn.Linear(dim_in, dim_head * num_head, bias=False)
        self.W_K = nn.Linear(dim_in, dim_head * num_head, bias=False)
        self.W_V = nn.Linear(dim_in, dim_head * num_head, bias=False)
        self.fc = nn.Linear(dim_head * num_head, dim_in, bias=False)
        self.layer_norm = nn.LayerNorm(dim_in)
        self.attention = ScaledDotProductAttention()

    def forward(self, x):
        assert x.ndim == 3
        batch_size = x.size(0)
        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)
        Q = Q.view(batch_size * self.num_head, -1, self.dim_head)
        K = K.view(batch_size * self.num_head, -1, self.dim_head)
        V = V.view(batch_size * self.num_head, -1, self.dim_head)
        scale = K.size(-1)**-0.5  # 缩放因子
        context = self.attention(Q, K, V, scale)
        context = context.view(batch_size, -1, self.dim_head * self.num_head)
        out = self.fc(context)
        # out = self.dropout(out)
        out = out + x
        out = self.layer_norm(out)
        return out


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, in_dim, feat_dim):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = nn.Sequential(nn.Linear(in_dim, feat_dim, bias=False),
                                nn.ReLU(),
                                nn.Linear(feat_dim, in_dim, bias=False))
        self.ln = nn.LayerNorm(in_dim)

    def forward(self, inputs):
        residual = inputs
        output = self.fc(inputs)
        return self.ln(output + residual)


class EncoderLayer(nn.Module):
    def __init__(self, dim_in, num_head, dim_hidden):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(dim_in, num_head)
        self.pos_ffn = PoswiseFeedForwardNet(dim_in, dim_hidden)

    def forward(self, x):
        out = self.attention(x)
        out = self.pos_ffn(out)
        return out

@BACKBONE.register_module()
class TransformerEncoder(nn.Module):
    def __init__(self, dim_in, num_head, dim_hidden, num_layers, dim_out, seq_len):
        super(TransformerEncoder, self).__init__()
        self.encoder = nn.ModuleList([
            EncoderLayer(dim_in, num_head, dim_hidden)
            for _ in range(num_layers)
        ])
        self.fc = nn.Linear(seq_len * dim_in, dim_out)
        self.dropout = nn.Dropout(0.8)

    def forward(self, x, meta_info):
        for encoder in self.encoder:
            x = encoder(x)
        x = x.view(x.shape[0], -1)
        x = self.dropout(x)
        return self.fc(x)


# @BACKBONE.register_module()
# class SelfAttnSingleHead(nn.Module):
#     def __init__(self, dim_in, num_head, dim_hidden, num_layers):
#         super(SelfAttnSingleHead, self).__init__()
#         self.encoder = TransformerEncoder(dim_in, num_head, dim_hidden,
#                                           num_layers)
#
#     def forward(self, x):
#         return self.encoder(x)
#
#     def forward_train(self, x, gt_labels):
#         activation = self(x)
#         return self.cls_head.forward_train(activation, gt_labels)
#
#     def simple_test(self, x):
#         activation = self(x)
#         return self.cls_head.simple_test(activation)


if __name__ == '__main__':
    model = TransformerEncoder(256, 5, 1024, 3)
    inputs_x = torch.randn((4, 256, 1024))
    inputs_y = torch.randn((4, 300, 512))
    model(inputs_x)
