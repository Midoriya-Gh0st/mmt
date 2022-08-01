import torch
from torch import nn
import torch.nn.functional as F

from modules.transformer import TransformerEncoder


class MULTModel(nn.Module):
    def __init__(self, hyp_params):
        """
        Construct a MulT model.
        """
        super(MULTModel, self).__init__()
        self.orig_d_l, self.orig_d_a, self.orig_d_v = hyp_params.orig_d_l, hyp_params.orig_d_a, hyp_params.orig_d_v  # 300, 5, 20
        self.d_l, self.d_a, self.d_v = 40, 40, 40
        self.vonly = hyp_params.vonly
        self.aonly = hyp_params.aonly
        self.lonly = hyp_params.lonly
        self.num_heads = hyp_params.num_heads
        self.layers = hyp_params.layers
        self.attn_dropout = hyp_params.attn_dropout
        self.attn_dropout_a = hyp_params.attn_dropout_a
        self.attn_dropout_v = hyp_params.attn_dropout_v
        self.relu_dropout = hyp_params.relu_dropout
        self.res_dropout = hyp_params.res_dropout
        self.out_dropout = hyp_params.out_dropout
        self.embed_dropout = hyp_params.embed_dropout
        self.attn_mask = hyp_params.attn_mask

        self.w_a_v = nn.parameter.Parameter(torch.ones(2), requires_grad=True)
        self.w_comb = nn.parameter.Parameter(torch.ones(2), requires_grad=True)

        combined_dim = self.d_l + self.d_a + self.d_v

        self.partial_mode = self.lonly + self.aonly + self.vonly
        if self.partial_mode == 1:
            combined_dim = 2 * self.d_l  # assuming d_l == d_a == d_v
        else:
            combined_dim = 2 * (self.d_l + self.d_a + self.d_v)
            combined_dim = 2 * (self.d_a + self.d_v)

        output_dim = hyp_params.output_dim  # This is actually not a hyperparameter :-)

        # 1. Temporal convolutional layers
        self.proj_l = nn.Conv1d(self.orig_d_l, self.d_l, kernel_size=1, padding=0, bias=False)
        self.proj_a = nn.Conv1d(self.orig_d_a, self.d_a, kernel_size=1, padding=0, bias=False)
        self.proj_v = nn.Conv1d(self.orig_d_v, self.d_v, kernel_size=1, padding=0, bias=False)

        """ ---------------------------- """
        """ add-lstm """
        # TODO: cLSTM() 实现? 和现在 Conv+LSTM 一样吗?
        num_layers = 2
        # BF = False  # is_batch_first?
        BD = False  # is_bidirectional? # 设置成True, 后面会有 40*2=80, 还是应该取后面[:40]?
        # seq_len 不包含在内
        in_size = 40
        out_size = 40
        BIAS = False  # is_bias_open?
        self.gru_l = nn.GRU(in_size, out_size, num_layers, bidirectional=BD, bias=BIAS)  # 需求: [input_size, h_size, num_layers]
        self.gru_a = nn.GRU(in_size, out_size, num_layers, bidirectional=BD, bias=BIAS)  #
        self.gru_v = nn.GRU(in_size, out_size, num_layers, dropout=0, bidirectional=BD, bias=BIAS)  # default-setting

        # 2. Crossmodal Attentions
        if self.lonly:
            self.trans_l_with_a = self.get_network(self_type='la')
            self.trans_l_with_v = self.get_network(self_type='lv')
        if self.aonly:
            self.trans_a_with_l = self.get_network(self_type='al')
            self.trans_a_with_v = self.get_network(self_type='av')
        if self.vonly:
            self.trans_v_with_l = self.get_network(self_type='vl')
            self.trans_v_with_a = self.get_network(self_type='va')

        # 3. Self Attentions (Could be replaced by LSTMs, GRUs, etc.)
        #    [e.g., self.trans_x_mem = nn.LSTM(self.d_x, self.d_x, 1)
        self.trans_l_mem = self.get_network(self_type='l_mem', layers=3)
        self.trans_a_mem = self.get_network(self_type='a_mem', layers=3)
        self.trans_v_mem = self.get_network(self_type='v_mem', layers=3)

        # Projection layers
        self.proj1 = nn.Linear(combined_dim, combined_dim)
        self.proj2 = nn.Linear(combined_dim, combined_dim)
        self.out_layer = nn.Linear(combined_dim, output_dim)

    def get_network(self, self_type='l', layers=-1):
        if self_type in ['l', 'al', 'vl']:
            embed_dim, attn_dropout = self.d_l, self.attn_dropout
        elif self_type in ['a', 'la', 'va']:
            embed_dim, attn_dropout = self.d_a, self.attn_dropout_a
        elif self_type in ['v', 'lv', 'av']:
            embed_dim, attn_dropout = self.d_v, self.attn_dropout_v
        elif self_type == 'l_mem':
            embed_dim, attn_dropout = 2 * self.d_l, self.attn_dropout
        elif self_type == 'a_mem':
            embed_dim, attn_dropout = 2 * self.d_a, self.attn_dropout
        elif self_type == 'v_mem':
            embed_dim, attn_dropout = 2 * self.d_v, self.attn_dropout
        else:
            raise ValueError("Unknown network type")

        return TransformerEncoder(embed_dim=embed_dim,
                                  num_heads=self.num_heads,
                                  layers=max(self.layers, layers),
                                  attn_dropout=attn_dropout,
                                  relu_dropout=self.relu_dropout,
                                  res_dropout=self.res_dropout,
                                  embed_dropout=self.embed_dropout,
                                  attn_mask=self.attn_mask)

    def forward(self, x_l, x_a, x_v):
        """
        text, audio, and vision should have dimension [batch_size, seq_len, n_features]
        """
        # input(f'[stat-1]: {x_l.shape, x_a.shape, x_v.shape}')  # [bsz, 300/5/20]
        x_l = F.dropout(x_l.transpose(1, 2), p=self.embed_dropout, training=self.training)
        x_a = x_a.transpose(1, 2)
        x_v = x_v.transpose(1, 2)

        # Project the textual/visual/audio features
        proj_x_l = x_l if self.orig_d_l == self.d_l else self.proj_l(x_l)
        proj_x_a = x_a if self.orig_d_a == self.d_a else self.proj_a(x_a)
        proj_x_v = x_v if self.orig_d_v == self.d_v else self.proj_v(x_v)
        # print(f"[-1-]: {proj_x_l.shape, proj_x_a.shape, proj_x_v.shape}")  # [bsz, 40, tgt_len]

        proj_x_a = proj_x_a.permute(2, 0, 1)
        proj_x_v = proj_x_v.permute(2, 0, 1)
        proj_x_l = proj_x_l.permute(2, 0, 1)
        # input(f"[-2-]: {proj_x_l.shape, proj_x_a.shape, proj_x_v.shape}")

        """ -------------------------- """
        # 添加GRU, 再次训练;
        # in_size == hidden_size?
        proj_x_l, f_l = self.gru_l(proj_x_l)
        proj_x_a, f_a = self.gru_a(proj_x_a)
        proj_x_v, f_v = self.gru_v(proj_x_v)
        # input(f"[ck-hidden]: {f_l.shape, f_a.shape, f_v.shape}")  # ([2, 2, 40]): [layers, bsz, hidden], 后面可能用到吗?
        # input(f"[-3-]: {proj_x_l.shape, proj_x_a.shape, proj_x_v.shape}")
        # (torch.Size([50, 2, 40]), torch.Size([375, 2, 40]), torch.Size([500, 2, 40]))  # [seq_len, bsz, D*hidden]  # [50, 2, 40] # if bi-d, 80

        """ modals shape """
        # input(f"[feature-dim-shape]: {proj_x_l.shape, proj_x_a.shape, proj_x_v.shape}")
        # (torch.Size([50, 2, 40]), torch.Size([375, 2, 40]), torch.Size([500, 2, 40]))

        """ [Audio & Visual] """
        # audio:
        h_a_with_vs = self.trans_a_with_v(proj_x_a, proj_x_v, proj_x_v)  # torch.Size([375, 2, 40])
        last_h_a = h_a_with_vs[-1]  # 取最后一层进行cat;

        # visual:
        h_v_with_as = self.trans_v_with_a(proj_x_v, proj_x_a, proj_x_a)  # torch.Size([500, 2, 40])
        last_h_v = h_v_with_as[-1]

        """ 加上后, 性能有一点下降 """
        # 加上 RNN 的最后
        # last_h_a += f_a[-1]
        # last_h_v += f_v[-1]

        comb_a_v = torch.cat([last_h_a, last_h_v], dim=-1)
        # comb_a_v = self.trans_v_mem(comb_a_v)
        # print("comb-shape", comb_a_v.shape)

        # assert list(comb_a_v.shape) == [2, self.d_l*2]

        """ 上面的两个再与 [Text] 进行 trans 呢? """

        # 合并 [A + V], 通过projection, 弥合模态间features_dim的不同;

        """ [Text] """
        # 拼接hidden_feature, 还是融合raw_feature?
        # 要有相同的seq_len;
        # print("[1-0]", proj_x_l.shape)
        h_l_with_a = self.trans_l_with_a(proj_x_l, h_a_with_vs, h_a_with_vs)  # Dimension (L, N, d_l)
        # print("[1-1]", h_l_with_a.shape)
        # h_l_with_a_v = self.trans_l_with_a_v(h_l_with_a, proj_x_v, proj_x_v)  # Dimension (L, N, d_l)
        # print("[1-2]", h_l_with_a_v.shape)
        h_l_with_v = self.trans_l_with_v(proj_x_l, h_v_with_as, h_v_with_as)  # Dimension (L, N, d_l)
        # h_l_with_v_a = self.trans_l_with_v_a(h_l_with_v, proj_x_a, proj_x_a)  # Dimension (L, N, d_l)

        h_l_emph = torch.cat([h_l_with_a * self.w_a_v[0], h_l_with_v * self.w_a_v[1]], dim=-1)  # TODO: 添加[a-v modal]的weight
        # h_l_emph = torch.cat([h_l_with_a, h_l_with_v], dim=-1)  # 原先无weight
        h_l_emph = self.trans_l_mem(h_l_emph)
        # print("[1-4]", h_l_emph.shape)
        last_h_l = h_l_emph[-1]
        # assert list(last_h_l.shape) == [2, self.d_l * 2]

        # TODO: 考虑 co-attn, 将[A/V]叠加, 然后进行 self-attn;

        last_hs = torch.cat([last_h_l * self.w_comb[0], comb_a_v * self.w_comb[1]], dim=1)  # TODO: 添加[text - comb]的weight
        # last_hs = torch.cat([last_h_l, comb_a_v], dim=1)
        # print(last_h_l.shape)  # [8, 80]
        # print(comb_a_v.shape)  # [8, 80]
        # print(last_hs.shape)   # [8, 160]
        # input()

        # A residual block
        last_hs_proj = self.proj2(F.dropout(F.relu(self.proj1(last_hs)), p=self.out_dropout, training=self.training))
        last_hs_proj += last_hs

        output = self.out_layer(last_hs_proj)
        return output, last_hs





