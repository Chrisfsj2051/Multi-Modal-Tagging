import numpy as np
import torch
import torch.nn as nn

def get_attn_pad_mask(seq_q, seq_k):
    '''
    seq_q: [batch_size, seq_len]
    seq_k: [batch_size, seq_len]
    seq_len could be src_len or it could be tgt_len
    seq_len in seq_q and seq_len in seq_k maybe not equal
    '''
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    # eq(zero) is PAD token
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # [batch_size, 1, len_k], False is masked
    return pad_attn_mask.expand(batch_size, len_q, len_k)  # [batch_size, len_q, len_k]

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k

    def forward(self, Q, K, V, attn_mask):
        '''
        Q: [batch_size, n_heads, len_q, d_k]
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v(=len_k), d_v]
        attn_mask: [batch_size, n_heads, seq_len, seq_len]
        '''
        # scores : [batch_size, n_heads, len_q, len_k]
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)
        # Fills elements of self tensor with value where mask is True.
        scores.masked_fill_(attn_mask, -1e9)
        attn = nn.Softmax(dim=-1)(scores)
        # [batch_size, n_heads, len_q, d_v]
        context = torch.matmul(attn, V)
        return context, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_k, d_v, n_heads, in_dim_k, in_dim_v):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(in_dim_k, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(in_dim_v, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(in_dim_v, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, in_dim_k, bias=False)
        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v
        self.ln = nn.LayerNorm(in_dim_k)
        self.dot_attn = ScaledDotProductAttention(in_dim_k)

    def forward(self, input_Q, input_K, input_V, attn_mask):
        '''
        input_Q: [batch_size, len_q, d_model]
        input_K: [batch_size, len_k, d_model]
        input_V: [batch_size, len_v(=len_k), d_model]
        attn_mask: [batch_size, seq_len, seq_len]
        '''
        residual, batch_size = input_Q, input_Q.size(0)
        # (B, S, D) -proj-> (B, S, D_new) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        Q = self.W_Q(input_Q).view(
            batch_size, -1, self.n_heads,
            self.d_k).transpose(1, 2)  # Q: [batch_size, n_heads, len_q, d_k]
        K = self.W_K(input_K).view(
            batch_size, -1, self.n_heads,
            self.d_k).transpose(1, 2)  # K: [batch_size, n_heads, len_k, d_k]
        V = self.W_V(input_V).view(
            batch_size, -1, self.n_heads, self.d_v).transpose(
                1, 2)  # V: [batch_size, n_heads, len_v(=len_k), d_v]
        # attn_mask : [batch_size, n_heads, seq_len, seq_len]
        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1,1)
        # context: [batch_size, n_heads, len_q, d_v], attn: [batch_size, n_heads, len_q, len_k]
        context, attn = self.dot_attn(Q, K, V, attn_mask)
        context = context.transpose(1, 2).reshape(
            batch_size, -1, self.n_heads *
            self.d_v)  # context: [batch_size, len_q, n_heads * d_v]
        output = self.fc(context)  # [batch_size, len_q, d_model]
        return self.ln(output + residual), attn


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, in_dim, feat_dim):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = nn.Sequential(nn.Linear(in_dim, feat_dim, bias=False),
                                nn.ReLU(),
                                nn.Linear(feat_dim, in_dim, bias=False))
        self.ln = nn.LayerNorm(in_dim)

    def forward(self, inputs):
        '''
        inputs: [batch_size, seq_len, d_model]
        '''
        residual = inputs
        output = self.fc(inputs)
        return self.ln(output + residual)  # [batch_size, seq_len, d_model]


class EncoderLayer(nn.Module):
    def __init__(self, d_k, d_v, n_heads, in_dim_x, in_dim_y, ff_feat_dim):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention(d_k, d_v, n_heads, in_dim_x, in_dim_y)
        self.pos_ffn = PoswiseFeedForwardNet(in_dim_x, ff_feat_dim)

    def forward(self, x, y, attn_mask):
        '''
        enc_inputs: [batch_size, src_len, d_model]
        enc_self_attn_mask: [batch_size, src_len, src_len]
        '''
        # enc_outputs: [batch_size, src_len, d_model], attn: [batch_size, n_heads, src_len, src_len]
        # enc_inputs to same Q,K,V
        enc_outputs, attn = self.enc_self_attn(x, y, y, attn_mask)
        # enc_outputs: [batch_size, src_len, d_model]
        enc_outputs = self.pos_ffn(enc_outputs)
        return enc_outputs, attn


class BiAttnFusion(nn.Module):
    def __init__(self, d_k, d_v, n_heads, in_dim_x, in_dim_y, n_layers, ff_feat_dim):
        super(BiAttnFusion, self).__init__()
        self.attn_x = nn.ModuleList([
            EncoderLayer(d_k, d_v, n_heads, in_dim_x, in_dim_y, ff_feat_dim)
            for _ in range(n_layers)
        ])
        self.attn_y = nn.ModuleList([
            EncoderLayer(d_k, d_v, n_heads, in_dim_y, in_dim_x, ff_feat_dim)
            for _ in range(n_layers)
        ])

    def forward(self, x, y):
        '''
        enc_inputs: [batch_size, src_len]
        '''
        assert x.shape[1] == y.shape[1]
        # seq_len = max(x.shape[1], y.shape[1])
        # assert x.shape[1] <= seq_len
        # x = torch.cat([x, torch.zeros_like(x)[:, :seq_len-x.shape[1]]], 1)
        attn_pad_x = get_attn_pad_mask(x[:, :, 0], y[:, :, 0])
        attn_pad_y = get_attn_pad_mask(y[:, :, 0], x[:, :, 0])
        for attn_x, attn_y in zip(self.attn_x, self.attn_y):
            # enc_outputs: [batch_size, src_len, d_model], enc_self_attn: [batch_size, n_heads, src_len, src_len]
            x_ = attn_x(x, y, attn_pad_x)[0]
            y_ = attn_y(y, x, attn_pad_y)[0]
            x, y = x_, y_

        return x, y


if __name__ == '__main__':
    model = BiAttnFusion(d_k=64,
                         d_v=64,
                         n_layers=6,
                         in_dim_x=1024,
                         in_dim_y=512,
                         n_heads=8,
                         ff_feat_dim=2048)
    inputs_x = torch.randn((4, 256, 1024))
    inputs_y = torch.randn((4, 300, 512))
    model(inputs_x, inputs_y)
