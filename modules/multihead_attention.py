import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F
import pickle as pkl
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

        """ 尝试 head weight  """
        # self.head_linear = nn.Linear(self.num_heads, 1)  # 多元回归, in_size=num_heads, out_size=1; (各head的权重)
        self.w_head = torch.nn.parameter.Parameter(torch.ones(self.num_heads), requires_grad=True)
        # print(self.w_head)

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

        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if k is not None:
            k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if v is not None:
            v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        src_len = k.size(1)

        if self.add_zero_attn:
            src_len += 1
            k = torch.cat([k, k.new_zeros((k.size(0), 1) + k.size()[2:])], dim=1)
            v = torch.cat([v, v.new_zeros((v.size(0), 1) + v.size()[2:])], dim=1)
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)

        attn_weights = torch.bmm(q, k.transpose(1, 2))
        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        if attn_mask is not None:
            try:
                attn_weights += attn_mask.unsqueeze(0)
            except:
                print(attn_weights.shape)
                print(attn_mask.unsqueeze(0).shape)
                assert False

        # attn_weights = F.softmax(attn_weights.float(), dim=-1).type_as(attn_weights)
        # # attn_weights = F.relu(attn_weights)
        # # attn_weights = attn_weights / torch.max(attn_weights)
        # attn_weights = F.dropout(attn_weights, p=self.attn_dropout, training=self.training)
        # print(f"[check-attn-weight shape]: {attn_weights.shape}")  # [2 * 10, 50, 375] = [bsz * heads, ]
        #
        # attn = torch.bmm(attn_weights, v)
        # print(f"[check-attn shape]: {attn.shape}")
        # input()
        # assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]  # [20, 50, 4]
        #
        # attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        # attn = self.out_proj(attn)

        def cur_show(ws, title="null"):
            ws = ws.cpu().clone().detach().numpy()
            plt.matshow(ws)
            plt.title(title)
            plt.show()

        attn_weights = F.softmax(attn_weights.float(), dim=-1).type_as(attn_weights)
        # attn_weights = F.relu(attn_weights)
        # attn_weights = attn_weights / torch.max(attn_weights)
        attn_weights = F.dropout(attn_weights, p=self.attn_dropout, training=self.training)
        # print(f"[check-attn-weight shape]: {attn_weights.shape}")  # [2 * 10, 50, 375] = [bsz * heads, tgt_len, src_len]
        # print(f"attn_w_[0] (sample-1-h-1):\n", attn_weights[0])
        # cur_show(attn_weights[0])

        # compute heads correlation
        attn = torch.bmm(attn_weights, v)
        # print(f"[check-attn shape]: {attn.shape}")
        # input()
        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]  # [20, 50, 4]  # head_dim, 每个head平均从[all_dim获取一定的head_dim];

        # ----------------------------------------
        # head-proc
        attn1 = attn.reshape(bsz, self.num_heads, tgt_len, self.head_dim).transpose(1, -1)  # [bsz, tgt_len, head_dim, num_heads]

        # TODO: place in _init_();
        # whole_heads = self.head_linear(attn1)
        # print("whole-heads:", whole_heads.shape, "\n", whole_heads)  # [2, 4, 50, 1]

        # normalize weight
        # heads_weight = F.softmax(self.head_linear.weight)  # 10, num_heads
        # heads_weight = heads_weight / heads_weight.max()
        # weight = heads_weight.unsqueeze(2).unsqueeze(3)  # [1, num_heads] => [1, num_heads, 1, 1], broadcast

        # print("shape-1:", attn.shape)
        # print("shape-2:", weight.shape)

        # weighted attn
        # TODO: 不同batch用相同的head_w? 每个sample用单独的weight?
        # test-1: 每个sample共享head_weight;
        attn = attn.reshape(bsz, self.num_heads, tgt_len, self.head_dim)
        head_weight = self.w_head.reshape(1, -1, 1, 1)
        attn = attn * head_weight  # 对attn, 按head, 进行weight;

        # ----------------------------------

        # 注意这里, shape
        # 原始代码: [bsz * self.num_heads, tgt_len, self.head_dim]  =>
        # attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)  # 这里进行了[heads]合并

        # head_weight: [bsz, self.num_heads, tgt_len, self.head_dim]
        attn = attn.transpose(1, 2)  # => [bsz, tgt_len, self.num_heads, self.head_dim]
        # print("[attn-1]:", attn.shape)
        attn = attn.transpose(0, 1)  # => [tgt_len, bsz, self.num_heads, self.head_dim]
        # print("[attn-2]:", attn.shape)
        attn = attn.contiguous().view(tgt_len, bsz, embed_dim)  # TODO: 什么时候使用 contiguous, 缩减?
        # print("[attn-3]:", attn.shape)
        # input()
        #

        attn = self.out_proj(attn)

        def pkl_encoder(var, file_name):
            # pickle a variable to a file
            file = open(file_name, 'wb')
            pkl.dump(var.cpu().clone(), file)
            file.close()
            print("save ok!")
            input()

        # pkl_encoder(attn_weights, "mosi_attn.pkl")

        return attn, attn_weights

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
