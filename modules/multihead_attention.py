import math

import numpy as np
import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F
import sys
import matplotlib.pyplot as plt
import numpy.ma as ma

# Code adapted from the fairseq repo.


def cur_show(ws, title="null"):
    ws = ws.cpu().clone()
    plt.matshow(ws)
    plt.title(title)
    plt.show()


def get_val_lens(p_x):
    """ compute the real seq_len. """
    # p_x = p_x.transpose(0, 1)  # [b_s, len, dim]
    kp_mask = (p_x != 0).float()  # create mask tensor
    real_lens = kp_mask.count_nonzero(dim=2).count_nonzero(dim=1)
    return real_lens


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

    def forward(self, query, key, value, attn_mask=None):  # 新增 padding
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

        def check_min(w, descp):
            # 检查attn_weight的最大值, 最小值
            print(f"[{descp}] min:", w.min())

        if self.add_zero_attn:  # not used
            print("add_zero_attn")
            src_len += 1
            k = torch.cat([k, k.new_zeros((k.size(0), 1) + k.size()[2:])], dim=1)
            v = torch.cat([v, v.new_zeros((v.size(0), 1) + v.size()[2:])], dim=1)
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)

        """ attn_weights compute. """
        attn_weights = torch.bmm(q, k.transpose(1, 2))
        # print("size:", attn_weights.size())  # torch.Size([40, 50, 375])

        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        # ---
        # 检查 mask 是否 正确.
        def check_mask(my_mask):
            mask_l_a = my_mask
            print("shape:", mask_l_a.shape)
            print(mask_l_a[0])
            print("A:", get_val_lens(mask_l_a))
            print("B", get_val_lens(mask_l_a.transpose(1, 2)))
            cur_show(mask_l_a[0])
            cur_show(mask_l_a[4])
            cur_show(mask_l_a[5])
            cur_show(mask_l_a[10])
            cur_show(mask_l_a[15])
            cur_show(mask_l_a[20])
            cur_show(mask_l_a[25])

        # check_mask(attn_mask)
        # ---
        def apply_mask(w, mask, zero_mask=False):
            """ process seq. len. problem. """
            if mask is not None:
                try:
                    # print("attn_mask.shape:", attn_mask.shape)
                    # w = w * mask.repeat(self.num_heads, 1, 1).squeeze()  # 不能使用 "*=".
                    mask = mask.logical_not()  # 取not, 使mask中为True的位置表示需要被mask;
                    # check_mask(mask)  # mask: ok
                    # print("repeat-0", mask.shape)

                    mask = mask.logical_not().repeat_interleave(self.num_heads, dim=0).squeeze()  # 散开heads
                    # print("repeat-1", mask.shape)
                    # 之前(repeat): # repeat之后, 是一轮再一轮, 不是把相同的放在一起;
                    # 现在: """ 要把相同sample的heads放在一起 """
                    # check_mask(mask)
                    # check_mask(w)

                    if zero_mask:
                        w = w.masked_fill(mask.logical_not(), 0)
                    else:
                        # w[mask == 0] = -1e9
                        w = w.masked_fill(mask.logical_not(), -1e9)

                    # check_mask(w)

                    return w  # attn_weights
                except:
                    print(attn_weights.shape)
                    print(mask.repeat(self.num_heads, 1, 1).unsqueeze(0).shape)
                    assert False
            else:
                # print("[MHA]apply: mask is none.")
                return w

        # check_mask(attn_weights)

        # mask-1, 因为之后使用softmax.
        # mask with -inf (-1e9).
        attn_weights = apply_mask(attn_weights, attn_mask)  # attn_mask: BoolMask tensor
        # input("[1] check.")
        # print("mask:", attn_mask)
        # print(attn_weights[-1])
        # input("[1] end.")

        attn_weights = F.softmax(attn_weights.float(), dim=-1).type_as(attn_weights)
        # input("[2] check.")
        # print(attn_weights[-1])
        # input("[2] end.")

        # mask with zero.
        attn_weights = apply_mask(attn_weights, attn_mask, zero_mask=True)
        # input("[3] check.")
        # print(attn_weights[-1])
        # input("[3] end.")

        # 有weight泄露, 经过softmax, pad部分也被观测了.
        # 即使把额外的部分变成0也不对, 因为最小值有<0的.
        # 解决方案: 使用-math.inf

        # attn_weights = F.relu(attn_weights)
        # attn_weights = attn_weights / torch.max(attn_weights)
        attn_weights = F.dropout(attn_weights, p=self.attn_dropout, training=self.training)

        """ weight对attn产生影响的地方在此处! """
        # 在计算attn之前, 对attn_weight进行处理(prune)
        # 为什么要mask? 使其为0, 不会被梯度更新.

        attn = torch.bmm(attn_weights, v)
        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]

        # print("embd_dim:", self.embed_dim)  # 30
        # print("head_dim:", self.head_dim)   # 10

        attn = attn.transpose(0, 1).contiguous().reshape(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)

        # average attention weights over heads
        attn_weights = attn_weights.reshape(bsz, self.num_heads, tgt_len, src_len)

        # --- 保存
        # import pickle
        # def pkl_encoder(var, file_name):
        #     # pickle a variable to a file
        #     file = open(file_name, 'wb')
        #     pickle.dump(var, file)
        #     file.close()
        #     print("save ok!")
        # x = input("保存Layer-3:?")
        # if x == "ok":
        #     pkl_encoder(attn_weights, "mosi-attn-w-l3-bs4.pkl")
        #
        # t1 = get_val_lens(attn_weights[:, 0, :, :])
        # t2 = get_val_lens(attn_weights.transpose(-1, -2)[:, 0, :, :])
        # print("check len:", t1, t2)
        # ---

        attn_weights = attn_weights.sum(dim=1) / self.num_heads
        # input(f"The weight:{attn_weights.shape}")  # torch.Size([35, 50, 375])
        # cur_show(attn_weights[0])  # 第一个batch的attn_weights

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
