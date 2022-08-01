import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.parameter as P
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

        """ Modal weight """
        # emph layer weight
        self.w_emph_t = P.Parameter(torch.ones(2), requires_grad=True)       # for Text-with[A, V]
        self.w_emph_v = P.Parameter(torch.ones(2), requires_grad=True)       # for Visual-with-[L, A]
        self.w_emph_a = P.Parameter(torch.ones(2), requires_grad=True)       # for Audio-with-[V, A]

        # combine layer weight
        self.w_comb_t = P.Parameter(torch.ones(2), requires_grad=True)       # 用在 emph-t 与 [A, V] 之间;
        self.w_comb_v = P.Parameter(torch.ones(2), requires_grad=True)
        self.w_comb_a = P.Parameter(torch.ones(2), requires_grad=True)

        # final concat layer weight
        self.w_last = P.Parameter(torch.ones(3), requires_grad=True)         # t_comb, v_comb, a_comb, 之间的weight;

        # bias?
        # self.b_a_v = nn.parameter.Parameter(torch.zeros(1), requires_grad=True)
        # self.b_comb = nn.parameter.Parameter(torch.zeros(1), requires_grad=True)

        combined_dim = self.d_l + self.d_a + self.d_v

        self.partial_mode = self.lonly + self.aonly + self.vonly
        # input(f"partial_mode={self.partial_mode}")  # 3
        if self.partial_mode == 1:
            combined_dim = 3 * self.d_l  # assuming d_l == d_a == d_v
        else:
            combined_dim = 4 * (self.d_l + self.d_a + self.d_v)  # 2 * 120
            # input(f"comb_dim: {combined_dim}")
            # combined_dim = 2 * (self.d_a + self.d_v)

        output_dim = hyp_params.output_dim  # This is actually not a hyperparameter :-)

        # 1. Temporal convolutional layers
        self.proj_l = nn.Conv1d(self.orig_d_l, self.d_l, kernel_size=1, padding=0, bias=False)
        self.proj_a = nn.Conv1d(self.orig_d_a, self.d_a, kernel_size=1, padding=0, bias=False)
        self.proj_v = nn.Conv1d(self.orig_d_v, self.d_v, kernel_size=1, padding=0, bias=False)

        """ ---------------------------- """
        """ GRU """
        num_layers = 3
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
            self.trans_t_with_a = self.get_network(self_type='la')
            self.trans_t_with_v = self.get_network(self_type='lv')
        if self.aonly:
            self.trans_a_with_t = self.get_network(self_type='al')
            self.trans_a_with_v = self.get_network(self_type='av')
        if self.vonly:
            self.trans_v_with_t = self.get_network(self_type='vl')
            self.trans_v_with_a = self.get_network(self_type='va')

        # 3. Self Attentions (Could be replaced by LSTMs, GRUs, etc.)
        #    [e.g., self.trans_x_mem = nn.LSTM(self.d_x, self.d_x, 1)
        self.trans_t_mem = self.get_network(self_type='l_mem', layers=3)
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
        self.gru_l.flatten_parameters()
        self.gru_a.flatten_parameters()
        self.gru_v.flatten_parameters()
        
        # 添加GRU, 再次训练;
        # in_size == hidden_size?
        proj_x_l, f_l = self.gru_l(proj_x_l)
        proj_x_a, f_a = self.gru_a(proj_x_a)
        proj_x_v, f_v = self.gru_v(proj_x_v)
        # input(f"[ck-hidden]: {f_l.shape, f_a.shape, f_v.shape}")  # ([2, 2, 40]): [layers, bsz, hidden], 后面可能用到吗?
        # input(f"[-3-]: {proj_x_l.shape, proj_x_a.shape, proj_x_v.shape}")
        # (torch.Size([50, 2, 40]), torch.Size([375, 2, 40]), torch.Size([500, 2, 40]))  # [seq_len, bsz, D*hidden]  # [50, 2, 40] # if bi-d, 80
        # ----

        """ modals shape """
        # input(f"[feature-dim-shape]: {proj_x_l.shape, proj_x_a.shape, proj_x_v.shape}")
        # (torch.Size([50, 2, 40]), torch.Size([375, 2, 40]), torch.Size([500, 2, 40]))

        if self.lonly:
            """ Text & [Audio + Visual] """
            # audio:
            h_a_with_v = self.trans_a_with_v(proj_x_a, proj_x_v, proj_x_v)  # torch.Size([375, 2, 40])
            last_h_a_for_t = h_a_with_v[-1]  # 取最后一层进行cat;

            # visual:
            h_v_with_a = self.trans_v_with_a(proj_x_v, proj_x_a, proj_x_a)  # torch.Size([500, 2, 40])
            last_h_v_for_t = h_v_with_a[-1]

            # comb src to tgt
            comb_for_t = torch.cat([last_h_v_for_t, last_h_a_for_t], dim=-1)

            """ [Text] emph layer: use src modal to emph tgt modal """
            h_t_emph_a = self.trans_t_with_a(proj_x_l, h_a_with_v, h_a_with_v)  # Dimension (L, N, d_l)  # 同源代码一样;
            h_t_emph_v = self.trans_t_with_v(proj_x_l, h_v_with_a, h_v_with_a)  # Dimension (L, N, d_l)

            h_t_emph = torch.cat([h_t_emph_a * self.w_emph_t[0], h_t_emph_v * self.w_emph_t[1]], dim=-1)  # + self.b_a_v TODO: 尝试添加bias
            h_t_emph = self.trans_t_mem(h_t_emph)
            last_h_t_emph = h_t_emph[-1]

            """ combine layer """
            last_h_t = torch.cat([last_h_t_emph * self.w_comb_t[0], comb_for_t * self.w_comb_t[1]], dim=1)

        if self.vonly:
            """ Visual & [Text + Audio] """
            # text
            h_t_with_a = self.trans_t_with_a(proj_x_v, proj_x_a, proj_x_a)  # torch.Size([500, 2, 40])
            last_h_t_for_v = h_t_with_a[-1]

            # audio
            h_a_with_t = self.trans_a_with_t(proj_x_v, proj_x_a, proj_x_a)  # torch.Size([500, 2, 40])
            last_h_a_for_v = h_a_with_t[-1]

            # comb src to tgt
            comb_for_v = torch.cat([last_h_t_for_v, last_h_a_for_v], dim=-1)

            """ [Visual]: emph layer """
            h_v_emph_t = self.trans_v_with_t(proj_x_v, h_t_with_a, h_t_with_a)
            h_v_emph_a = self.trans_v_with_a(proj_x_v, h_a_with_t, h_a_with_t)

            h_v_emph = torch.cat([h_v_emph_t * self.w_emph_v[0], h_v_emph_a * self.w_emph_v[1]], dim=-1)
            h_v_emph = self.trans_v_mem(h_v_emph)
            last_h_v_emph = h_v_emph[-1]

            """ combine layer """
            last_h_v = torch.cat([last_h_v_emph * self.w_comb_v[0], comb_for_v * self.w_comb_v[1]], dim=1)

        if self.aonly:
            """ Audio & [Text + Visual] """
            # text
            h_t_with_v = self.trans_t_with_v(proj_x_v, proj_x_a, proj_x_a)  # torch.Size([500, 2, 40])
            last_h_t_for_a = h_t_with_v[-1]

            # visual
            h_v_with_t = self.trans_v_with_t(proj_x_v, proj_x_a, proj_x_a)  # torch.Size([500, 2, 40])
            last_h_v_for_a = h_v_with_t[-1]

            # comb src to tgt
            comb_for_a = torch.cat([last_h_t_for_a, last_h_v_for_a], dim=-1)

            """ [Audio]: emph layer """
            h_a_emph_t = self.trans_a_with_t(proj_x_l, h_t_with_v, h_t_with_v)
            h_a_emph_v = self.trans_a_with_v(proj_x_l, h_v_with_t, h_v_with_t)

            h_a_emph = torch.cat([h_a_emph_t * self.w_emph_a[0], h_a_emph_v * self.w_emph_a[1]], dim=-1)
            h_a_emph = self.trans_a_mem(h_a_emph)
            last_h_a_emph = h_a_emph[-1]

            """ combine layer """
            last_h_a = torch.cat([last_h_a_emph * self.w_comb_a[0], comb_for_a * self.w_comb_a[1]], dim=1)

        # if self.partial_mode == 3:
            # last_h = torch.cat([last_h_t * self.w_last[0], last_h_v * self.w_last[1], last_h_a * self.w_last[2]], dim=1)
        last_h = torch.cat([last_h_t, last_h_v, last_h_a], dim=1)

        # A residual block
        x1 = self.proj1(last_h)
        last_hs_proj = self.proj2(F.dropout(F.relu(x1), p=self.out_dropout, training=self.training))
        last_hs_proj += last_h

        output = self.out_layer(last_hs_proj)
        return output, last_h

        # TODO: prompt?
        """ 加上后, 性能有一点下降 """
        # 加上 RNN 的最后
        # last_h_a += f_a[-1]
        # last_h_v += f_v[-1]

        # 合并 [A + V], 通过projection, 弥合模态间features_dim的不同;

        """ descp: """
        # 拼接hidden_feature, 还是融合raw_feature?
        # 要有相同的seq_len;

        # TODO: 考虑 co-attn, 将[A/V]叠加, 然后进行 self-attn;







