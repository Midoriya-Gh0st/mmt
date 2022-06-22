import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from modules.transformer import TransformerEncoder
import wandb


class MULTModel(nn.Module):
    def __init__(self, hyp_params):
        """
        Construct a MulT model.
        """
        super(MULTModel, self).__init__()
        self.orig_d_l, self.orig_d_a, self.orig_d_v = hyp_params.orig_d_l, hyp_params.orig_d_a, hyp_params.orig_d_v
        self.d_l, self.d_a, self.d_v = 40, 40, 40
        self.vonly = hyp_params.vonly
        self.aonly = hyp_params.aonly
        self.lonly = hyp_params.lonly

        # self.num_heads = hyp_params.num_heads
        # self.layers = hyp_params.layers
        # self.attn_dropout = hyp_params.attn_dropout
        self.num_heads = wandb.config['num_heads']
        self.layers = wandb.config['nlevels']
        self.attn_dropout = wandb.config['attn_dropout']

        self.attn_dropout_a = hyp_params.attn_dropout_a
        self.attn_dropout_v = hyp_params.attn_dropout_v

        # self.out_dropout = hyp_params.out_dropout
        # self.embed_dropout = hyp_params.embed_dropout
        self.out_dropout = wandb.config['out_dropout']
        self.embed_dropout = wandb.config['embed_dropout']

        self.relu_dropout = hyp_params.relu_dropout
        self.res_dropout = hyp_params.res_dropout

        self.attn_mask = hyp_params.attn_mask
        combined_dim = self.d_l + self.d_a + self.d_v

        self.partial_mode = self.lonly + self.aonly + self.vonly
        if self.partial_mode == 1:
            combined_dim = 2 * self.d_l   # assuming d_l == d_a == d_v
        else:
            combined_dim = 2 * (self.d_l + self.d_a + self.d_v)
        
        output_dim = hyp_params.output_dim        # This is actually not a hyperparameter :-)

        # 1. Temporal convolutional layers
        self.proj_l = nn.Conv1d(self.orig_d_l, self.d_l, kernel_size=1, padding=0, bias=False)
        self.proj_a = nn.Conv1d(self.orig_d_a, self.d_a, kernel_size=1, padding=0, bias=False)
        self.proj_v = nn.Conv1d(self.orig_d_v, self.d_v, kernel_size=1, padding=0, bias=False)

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
            embed_dim, attn_dropout = 2*self.d_l, self.attn_dropout
        elif self_type == 'a_mem':
            embed_dim, attn_dropout = 2*self.d_a, self.attn_dropout
        elif self_type == 'v_mem':
            embed_dim, attn_dropout = 2*self.d_v, self.attn_dropout
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
        x_l = F.dropout(x_l.transpose(1, 2), p=self.embed_dropout, training=self.training)
        x_a = x_a.transpose(1, 2)
        x_v = x_v.transpose(1, 2)

        # check x_l / a / v, seq_len
        # print(x_l.shape, x_a.shape, x_v.shape)
        # torch.Size([2, 300, 50]) torch.Size([2, 74, 500]) torch.Size([2, 35, 500])

        # Project the textual/visual/audio features
        proj_x_l = x_l if self.orig_d_l == self.d_l else self.proj_l(x_l)
        proj_x_a = x_a if self.orig_d_a == self.d_a else self.proj_a(x_a)
        proj_x_v = x_v if self.orig_d_v == self.d_v else self.proj_v(x_v)
        proj_x_a = proj_x_a.permute(2, 0, 1)
        proj_x_v = proj_x_v.permute(2, 0, 1)
        proj_x_l = proj_x_l.permute(2, 0, 1)

        # print(proj_x_l.shape, proj_x_a.shape, proj_x_v.shape)
        # torch.Size([50, 2, 30]) torch.Size([500, 2, 30]) torch.Size([500, 2, 30])

        def get_val_lens(p_x):
            """ compute the real seq_len. """
            # p_x = p_x.transpose(0, 1)  # [b_s, len, dim]
            kp_mask = (p_x != 0).float()  # create mask tensor
            real_lens = kp_mask.count_nonzero(dim=2).count_nonzero(dim=1)
            return real_lens

        lens_l = get_val_lens(proj_x_l.transpose(0, 1))
        lens_a = get_val_lens(proj_x_a.transpose(0, 1))
        lens_v = get_val_lens(proj_x_v.transpose(0, 1))
        # print(lens_l, lens_a, lens_v)

        # input("check features:")
        # print(proj_x_l[0])  # 每个batch的前几行为0, 需要记住mask位置;

        # ---- my mask
        # T 与 [A, V], 之间的mask;
        # [b_s, tgt, src]

        def generate_mask(tgt_m, src_m, flag=0):
            # tgt_m: [50, 2, 30]  -> [2, 50, 500]
            # src_m: [500, 2, 30] -> [2, 50, 500]

            # print("src_m:", src_m.transpose(0, 1)[-1])
            lens_tgt = get_val_lens(tgt_m.transpose(0, 1))
            lens_src = get_val_lens(src_m.transpose(0, 1))

            tgt_len = tgt_m.shape[0]
            src_len = src_m.shape[0]

            # [1.1] tgt 第一列:
            # t = tgt_m.transpose(0, 1)  # [2, 50, 30]
            t = tgt_m.transpose(0, 1)[:, :, 0].unsqueeze(dim=-1)  # [2, 50, 1]  # 只取一列
            # print("check t:", t.shape)

            # [1.2] src 第一列:
            # s = src_m.transpose(0, 1)  # [2, 500, 30]
            s = src_m.transpose(0, 1)[:, :, 0].unsqueeze(dim=-1)  # [2, 500, 1]  # 只取一列

            # [2.1] tgt沿x轴延伸
            t = t.repeat(1, 1, src_len)  # [2, 50, 500]
            # print("[2] t.shape:", t.size())
            # print("[2] t[col-1]:", t[0, :, 0])  # 只有最后一部分是 real_value

            # [2.2] src沿x轴延伸
            s = s.repeat(1, 1, tgt_len)  # [2, 500, 50]
            # print("[2] s.shape:", s.size())
            # print("[2] s[col-1]:", s2:=s[0, :, 0])  # 只有最后一部分是 real_value
            # # print(get_val_lens(s2))

            # [3.1] src_m 转置
            s = s.transpose(1, 2)  # [2, 50, 500]

            # 有的src/tgt, 全为0;
            if 0 in lens_tgt and 0 not in lens_src:
                mask = s
                # print("0 in lens_tgt")
            elif 0 in lens_src and 0 not in lens_tgt:
                mask = t
                # print("0 in lens_src")
            elif 0 not in lens_tgt and 0 not in lens_src:
                # [3.2] 逻辑与
                # print(t.shape, s.shape)
                mask = torch.logical_and(t, s)
            else:
                mask = None
                # print("0 all.", lens_src, lens_tgt)
            # print("mask-shape:", mask.shape)
            #
            # if flag == 1:
            #     print("[check]: t[最后一个]")
            #     print(t[-1])
            #
            #     print("[check]: s[最后一个]")
            #     print(s[-1])

            return mask

        def cur_show(ws, title="null"):
            ws = ws.cpu().clone()
            plt.matshow(ws)
            plt.title(title)
            plt.show()
        # ----
        mask_l_a = generate_mask(proj_x_l, proj_x_a)

        """ 确认mask是否正确: 正确 """
        # print("shape:", mask_l_a.shape)
        # print(mask_l_a[0])
        # print(get_val_lens(mask_l_a))
        # print(get_val_lens(mask_l_a.transpose(1, 2)))
        # cur_show(mask_l_a[0])

        # input("check")
        mask_l_v = generate_mask(proj_x_l, proj_x_v, flag=1)
        mask_l_l = generate_mask(proj_x_l, proj_x_l)

        mask_a_l = generate_mask(proj_x_a, proj_x_l)
        mask_a_v = generate_mask(proj_x_a, proj_x_v)
        mask_a_a = generate_mask(proj_x_a, proj_x_a)

        mask_v_l = generate_mask(proj_x_v, proj_x_l)
        mask_v_a = generate_mask(proj_x_v, proj_x_a)
        mask_v_v = generate_mask(proj_x_v, proj_x_v)

        def check_lens(lens_x, mask):
            r = torch.equal(lens_x, get_val_lens(mask))
            if r is not True:
                # print("[lens error]", lens_x, get_val_lens(mask))  # tensor([18, 14, 19,  5]) tensor([18, 14, 19,  0])
                # print("lens_mask_l_v", get_val_lens(mask))
                # # print("[proj_x_l.shape", proj_x_l.shape)  # ([50, 4, 40])
                # test = proj_x_v.transpose(0, 1)
                # # cur_show(test[-1])
                # # cur_show(mask[-1])
                #
                # y1, y2, y3, y4 = test[3][-1], test[3][-2], test[3][-3], test[3][-4]
                # y5, y6 = test[3][-5], test[3][-6]
                # y7, y8 = test[3][-7], test[3][-8]
                # print(y1, y1.shape)
                # print(y2, y2.shape)
                # print(y3, y3.shape)
                # print(y4, y4.shape)
                # print(y5, y5.shape)
                # print(y6, y6.shape)
                # print(y7, y7.shape)
                # print(y8, y8.shape)

                # print(mask[3], mask[3].shape)

                # print("the mask:", mask[-1])
                # print("last 7 lens:", get_val_lens(proj_x_l.transpose(0, 1)[-7:]))
                # print(f"the proj_x_v:", proj_x_v[-7:])
                print(lens_l, lens_a, lens_v)
                # assert torch.equal(lens_x, get_val_lens(mask))
            return r

        def check_lens_func():
            #
            check_lens(lens_l, mask_l_a)
            check_lens(lens_a, mask_l_a.transpose(1, 2))

            # check_lens(lens_l, mask_l_v)  # ERR: tensor([18, 14, 19,  0])  # src(v), 全为0
            check_lens(lens_v, mask_l_v.transpose(1, 2))

            check_lens(lens_l, mask_l_l)
            check_lens(lens_l, mask_l_l.transpose(1, 2))

            check_lens(lens_a, mask_a_l)
            check_lens(lens_l, mask_a_l.transpose(1, 2))

            # check_lens(lens_a, mask_a_v)  # ERR:  tensor([53, 63, 69,  0])  # src(v), 全为0
            check_lens(lens_v, mask_a_v.transpose(1, 2))

            check_lens(lens_a, mask_a_a)
            check_lens(lens_a, mask_a_a.transpose(1, 2))

            check_lens(lens_v, mask_v_l)  # ERR:
            # check_lens(lens_l, mask_v_l.transpose(1, 2))  # tensor([18, 14, 19,  0])  # tgt(v), 全为0

            check_lens(lens_v, mask_v_a)  # ERR
            check_lens(lens_a, mask_v_a.transpose(1, 2))  # tensor([53, 63, 69,  0])  # tgt(v), 全为0

            check_lens(lens_v, mask_v_v)
            check_lens(lens_v, mask_v_v.transpose(1, 2))

        if self.lonly:
            # print("> [T]:")
            # # (V,A) --> L
            # print("A -> T:")                 # Q,        K,        V;
            h_l_with_as = self.trans_l_with_a(proj_x_l, proj_x_a, proj_x_a, attn_mask=mask_l_a)    # Dimension (L, N, d_l)
            # print(h_l_with_as)
            # input("A-T done.")
            #
            # print("V -> T:")
            h_l_with_vs = self.trans_l_with_v(proj_x_l, proj_x_v, proj_x_v, attn_mask=mask_l_v)    # Dimension (L, N, d_l)
            # input("V-T done.")

            h_ls = torch.cat([h_l_with_as, h_l_with_vs], dim=2)
            # print("T -> T:")
            h_ls = self.trans_l_mem(h_ls, attn_mask=mask_l_l)
            if type(h_ls) == tuple:
                h_ls = h_ls[0]
            last_h_l = last_hs = h_ls[-1]   # Take the last output for prediction
            # input("T-T done.")

        if self.aonly:
            # print("> [A]:")
            # # (L,V) --> A
            #
            # print("T -> A:")
            h_a_with_ls = self.trans_a_with_l(proj_x_a, proj_x_l, proj_x_l, attn_mask=mask_a_l)
            # input("T -> A done.")
            #
            # print("V -> A:")
            h_a_with_vs = self.trans_a_with_v(proj_x_a, proj_x_v, proj_x_v, attn_mask=mask_a_v)
            # input("V -> A done.")

            h_as = torch.cat([h_a_with_ls, h_a_with_vs], dim=2)
            # print("A -> A:")
            h_as = self.trans_a_mem(h_as, attn_mask=mask_a_a)
            if type(h_as) == tuple:
                h_as = h_as[0]
            last_h_a = last_hs = h_as[-1]
            # input("A -> A done.")

        if self.vonly:
            # print("> [V]:")
            # # (L,A) --> V
            # print("T -> V:")
            h_v_with_ls = self.trans_v_with_l(proj_x_v, proj_x_l, proj_x_l, attn_mask=mask_v_l)
            # input("T -> V done.")
            #
            # print("A -> V:")
            h_v_with_as = self.trans_v_with_a(proj_x_v, proj_x_a, proj_x_a, attn_mask=mask_v_a)
            # input("A -> V done.")

            h_vs = torch.cat([h_v_with_ls, h_v_with_as], dim=2)
            # print("V -> V:")
            h_vs = self.trans_v_mem(h_vs, attn_mask=mask_v_v)
            if type(h_vs) == tuple:
                h_vs = h_vs[0]
            last_h_v = last_hs = h_vs[-1]
            # input("V -> V done.")
        
        if self.partial_mode == 3:
            # print("> [All]:")
            last_hs = torch.cat([last_h_l, last_h_a, last_h_v], dim=1)
        
        # A residual block
        last_hs_proj = self.proj2(F.dropout(F.relu(self.proj1(last_hs)), p=self.out_dropout, training=self.training))
        last_hs_proj += last_hs
        
        output = self.out_layer(last_hs_proj)

        # print("output:", output.shape)    # torch.Size([35, 1])    # 最终情感评分
        # print("last_hs:", last_hs.shape)  # torch.Size([35, 180])

        return output, last_hs
