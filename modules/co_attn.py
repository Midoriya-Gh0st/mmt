import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F
import sys


# Code adapted from the fairseq repo.

class MultiheadAttention(nn.Module):
    """Multi-headed attention.
    See "Attention Is All You Need" for more details.
    """

    def __init__(self, embed_dim, num_heads, attn_dropout=0.,
                 bias=True, add_bias_kv=False, add_zero_attn=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.attn_dropout = attn_dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        self.in_proj_weight = Parameter(torch.Tensor(3 * embed_dim, embed_dim))
        self.register_parameter('in_proj_bias', None)
        if bias:
            self.in_proj_bias = Parameter(torch.Tensor(3 * embed_dim))
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        if add_bias_kv:
            self.bias_k = Parameter(torch.Tensor(1, 1, embed_dim))
            self.bias_v = Parameter(torch.Tensor(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self.reset_parameters()

        """ co-attn """
        co_bias = False
        self.tanh = nn.Tanh()
        self.co_dim = 10
        self.co_C_q_lin = nn.Linear(self.head_dim, self.head_dim, co_bias)  # W_b, [dim, dim], dim=4 = 40/10; 或者 [8];
        self.co_q_lin = nn.Linear(self.head_dim, self.co_dim, co_bias)
        self.co_k_lin = nn.Linear(self.head_dim, self.co_dim, co_bias)
        self.h_q_lin = nn.Linear(self.co_dim, 1, co_bias)
        self.h_k_lin = nn.Linear(self.co_dim, 1, co_bias)

        if self.head_dim == 8:
            # print(f"head-dim: {self.head_dim}")
            ...
        else:
            # print("dim=4")
            ...

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)

    def forward(self, query, key, value, attn_mask=None):
        """Input shape: Time x Batch x Channel
        Self-attention can be implemented by passing in the same arguments for
        query, key and value. Timesteps can be masked by supplying a T x T mask in the
        `attn_mask` argument. Padding elements can be excluded from
        the key by passing a binary ByteTensor (`key_padding_mask`) with shape:
        batch x src_len, where padding elements are indicated by 1s.
        """
        qkv_same = query.data_ptr() == key.data_ptr() == value.data_ptr()
        kv_same = key.data_ptr() == value.data_ptr()

        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        assert key.size() == value.size()

        # print("QKV-shape:", query.shape, key.shape)

        aved_state = None

        if qkv_same:
            # self-attention
            q, k, v = self.in_proj_qkv(query)
        elif kv_same:
            # encoder-decoder attention
            q = self.in_proj_q(query)

            if key is None:
                assert value is None
                k = v = None
            else:
                k, v = self.in_proj_kv(key)
        else:
            q = self.in_proj_q(query)
            k = self.in_proj_k(key)
            v = self.in_proj_v(value)
        q = q * self.scaling

        if self.bias_k is not None:
            assert self.bias_v is not None
            k = torch.cat([k, self.bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, self.bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)

        # qkv shape
        # print(f'[1] qkv-shape: {q.shape}, {k.shape}')  # torch.Size([375, 16, 40]), torch.Size([500, 16, 40])  # batch=16
        if self.head_dim == 8:
            # print(f'[2] qkv-shape: {q.shape}, {k.shape}')
            # torch.Size([50, 16, 80]), torch.Size([50, 16, 80])  # text-self-attn
            # input()
            ...

        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if k is not None:
            k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if v is not None:
            v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        else:
            print("V is None")
            if self.head_dim == 8:
                print("check-head-dim:", self.head_dim)
            else:
                print("not 8.")

        src_len = k.size(1)

        if self.add_zero_attn:
            print("[2]: self.add_zero_attn.")
            input()
            src_len += 1
            k = torch.cat([k, k.new_zeros((k.size(0), 1) + k.size()[2:])], dim=1)
            v = torch.cat([v, v.new_zeros((v.size(0), 1) + v.size()[2:])], dim=1)
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)
        else:
            # print(":no:")
            ...

        # input("[mha] (shape):")
        # print(f'q: {q.shape}; k: {k.shape}')  # q: torch.Size([160, 375, 4]); k: torch.Size([160, 500, 4])

        # -------------------
        """ co-attn """
        q_co = q
        k_co = k

        # print("[debug-1]", q.shape)  # torch.Size([160, 375, 4])

        q_W = self.co_C_q_lin(q)  # torch.Size([160, 375, 4])
        # print(f'q_co.shape: {q_W.shape}')
        C = torch.tanh(torch.bmm(q_W, k_co.transpose(1, 2)))  # [tgt_len, src_len]
        # bsz, num_heads = 16, 10  # TODO: 待定
        assert list(C.shape) == [bsz * self.num_heads, q_co.shape[1], k_co.shape[1]]
        # print("C-shape:", C.shape)  # [160, 375, 500]

        q_W = self.co_q_lin(q_co)
        k_W = self.co_k_lin(k_co)
        # print('[debug-2.1]:', q_co.shape, k_co.shape)  # torch.Size([160, 375, 4]) torch.Size([160, 500, 4])
        # print("[debug-2.2]:", q_W.shape, k_W.shape)  # [160, 375, 3] -> [bsz*num_heads, seq_len, co_dim]

        H_q_in = q_W.transpose(1, 2) + torch.bmm(k_W.transpose(1, 2), C.transpose(1, 2))  # [160, 500, 4] * [160, 375, 500] => [160, 4, 500] * [160, 500, 375]
        H_k_in = k_W.transpose(1, 2) + torch.bmm(q_W.transpose(1, 2), C)  # [160, 4, 375] * [160, 500, 375]
        # print("[debug-3]:", H_q_in.shape, H_k_in.shape)  # torch.Size([160, co_dim, seq_len])

        H_q = F.tanh(H_q_in)
        H_k = F.tanh(H_k_in)
        # print("[debug-4.1]: [H_{}]", H_q.shape, H_k.shape)  # torch.Size([160, co_dim, seq_len])  # [k, len]

        H_q_W = self.h_q_lin(H_q.transpose(1, 2))
        H_k_W = self.h_k_lin(H_k.transpose(1, 2))
        a_q = F.softmax(H_q_W.float(), dim=-1).type_as(H_q).repeat(1, 1, self.head_dim)
        a_k = F.softmax(H_k_W.float(), dim=-1).type_as(H_k).repeat(1, 1, self.head_dim)
        # print("[debug-4.2]: [a_{}]", a_q.shape, a_k.shape)  # k: torch.Size([160, 500, 4])
        assert list(a_q.shape) == [bsz * self.num_heads, tgt_len, self.head_dim]
        # TODO: 应该让 a_q 最后 为 "1" 吗? => 这其实是相当于 "sequence_attn", 对序列数据的attn; 即: position_attn; => 对序列的关注;

        # print("[debug-5]: [q, k]-1:", q.shape, k.shape)  # [160, 375, 4], [160, 500, 4]
        q = torch.mul(q, a_q)
        k = torch.mul(k, a_k)  # [bsz * num_heads, seq_len, head_dim]
        # print("[debug-6]: [q, k]-2:", q.shape, k.shape)
        assert list(q.shape) == [bsz * self.num_heads, tgt_len, self.head_dim]

        # ---------------------

        attn_weights = torch.bmm(q, k.transpose(1, 2))
        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        if attn_mask is not None:
            try:
                attn_weights += attn_mask.unsqueeze(0)
            except:
                print(attn_weights.shape)
                print(attn_mask.unsqueeze(0).shape)
                assert False

        attn_weights = F.softmax(attn_weights.float(), dim=-1).type_as(attn_weights)
        # attn_weights = F.relu(attn_weights)
        # attn_weights = attn_weights / torch.max(attn_weights)
        attn_weights = F.dropout(attn_weights, p=self.attn_dropout, training=self.training)

        attn = torch.bmm(attn_weights, v)

        """ 使用co-attn """
        # attn = q

        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]

        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)

        # average attention weights over heads
        attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
        attn_weights = attn_weights.sum(dim=1) / self.num_heads
        return attn, attn_weights

        # return attn, None

    def in_proj_qkv(self, query):
        return self._in_proj(query).chunk(3, dim=-1)

    def in_proj_kv(self, key):
        return self._in_proj(key, start=self.embed_dim).chunk(2, dim=-1)

    def in_proj_q(self, query, **kwargs):
        return self._in_proj(query, end=self.embed_dim, **kwargs)

    def in_proj_k(self, key):
        return self._in_proj(key, start=self.embed_dim, end=2 * self.embed_dim)

    def in_proj_v(self, value):
        return self._in_proj(value, start=2 * self.embed_dim)

    def _in_proj(self, input, start=0, end=None, **kwargs):
        weight = kwargs.get('weight', self.in_proj_weight)
        bias = kwargs.get('bias', self.in_proj_bias)
        weight = weight[start:end, :]
        if bias is not None:
            bias = bias[start:end]
        return F.linear(input, weight, bias)
